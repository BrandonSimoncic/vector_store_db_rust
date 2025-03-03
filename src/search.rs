use std::io::Error;
use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::Ollama;
use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::{Result, Tokenizer};
use serde_json;

#[derive(Debug)]
pub struct Filter {
    key: String,
    value: String,
}

impl Filter {
    pub fn new(key: String, value: String) -> Self {
        Filter { key, value }
    }
}
#[derive(Debug)]
pub struct Node {
    embedding: Vec<Vec<f32>>,
    sentence: String,
    text_id: usize,
    metadata: Filter,
}
impl Node{
    async fn new (words: String, id: usize, metadat: Filter, model: &String)-> Node{
        let embeddings = embed_ollama_data(model, &words).await.unwrap();
        Node {
            embedding: embeddings,
            sentence: words,
            text_id: id,
            metadata: metadat,
        }
    }
    
}
#[warn(dead_code)]
pub struct VectorStore {
    nodes: Vec<Node>,
    model: String,
}

impl VectorStore {
    pub fn new(m: &str) -> Self {
        VectorStore {
            nodes: Vec::new(),
            model: m.to_string(),
        }
    }
    //Get Embedding
    pub fn get(&self, text_id: usize) -> Option<&Vec<Vec<f32>>> {
        self.nodes.get(text_id).map(|node| &node.embedding)
    }
    //Add nodes to index
    pub async fn add(&mut self, words: String, metadata: Filter){
        let id = self.nodes.len() + 1;
        let node = Node::new(words, id, metadata, &self.model).await;
        self.nodes.push(node);
    }
    //Delete nodes using ref_doc_id
    pub fn delete(&mut self, ref_doc_id: usize){
        let item = self.nodes.iter().position(|r| r.text_id == ref_doc_id).unwrap();
        self.nodes.remove(item);
    }
    //Get nodes for response
    pub fn query(&self, query: &Node, filters: Vec<Filter>) -> Vec<&Node>{


        if self.nodes.len() == 0 {
           let empty =  Vec::new();
           return empty
        }
        let mut result_nodes = Vec::new();
        if filters.len() != 0 {
            result_nodes = filter_nodes(self, filters);
        }
        else {
            let (doc_embeddings, doc_ids): (Vec<_>, Vec<_>) = 
                self.nodes
                .iter()
                .map(|node| (node.embedding.clone(), node.text_id))
                .unzip();
            //this is probs not right
            let similarities_and_node_ids = get_top_k_embeddings(
                &query.embedding[0], 
                doc_embeddings[0].clone(), 
                doc_ids,
            5);


            result_nodes = similarities_and_node_ids
            .iter()
            .filter_map(|(id, _)| self.nodes.iter().find(|node| node.text_id == *id))
            .collect();
        }
        
        

    result_nodes
    }
    pub async fn search(&self, text: &str) -> Vec<&Node>{
        let chunks = parse_into_chunks(&text).unwrap();
        let mut result_nodes = Vec::new();
        for chunk in chunks{
            let embeddings = embed_ollama_data(&self.model, &chunk).await.unwrap();
            let query = Node {
                embedding: embeddings,
                sentence: chunk.to_string(),
                text_id: 0,
                metadata: Filter::new("".to_string(), "".to_string()),
            };
            let nodes = self.query(&query, Vec::new());
            result_nodes.extend(nodes);
        }
        result_nodes

    }
    pub async fn add_pdf(&mut self, path: String, metadata: Filter){
        let text = read_pdf(path);
        self.add(text, metadata).await;
        self.persist("data").unwrap();
    }
    //Persistent SimpleFectorStore to a dir
    fn persist(&self, dir: &str) -> std::result::Result<(), Error> {
        std::fs::create_dir_all(dir)?;

        let nodes_path = format!("{}/nodes.json", dir);
        let nodes_file = std::fs::File::create(nodes_path)?;
        let nodes_data: Vec<_> = self.nodes.iter().map(|node| {
            serde_json::json!({
                "embedding": node.embedding,
                "sentence": node.sentence,
                "text_id": node.text_id,
                "metadata": {
                    "key": node.metadata.key,
                    "value": node.metadata.value,
                }
            })
        }).collect();
        serde_json::to_writer(nodes_file, &nodes_data)?;

        let model_path = format!("{}/model.txt", dir);
        std::fs::write(model_path, &self.model)?;

        Ok(())
    }


}

fn filter_nodes(nodes: &VectorStore, filters: Vec<Filter>) -> Vec<&Node> {
    let mut filtered_nodes = Vec::new();

    for node in &nodes.nodes {
        let mut matches = true;
        for f in &filters {
            if f.key != node.metadata.key{
                matches = false;
                continue;
            }
            if f.value != node.metadata.value{
                matches = false;
                continue;
            }    
            if matches == true{ 
                filtered_nodes.push(node);
            }
        }
    }
    filtered_nodes
}
fn normalize(vector: &Vec<f32>) -> Vec<f32> {
    let norm = (vector.iter().map(|x| x * x).sum::<f32>()).sqrt();
    vector.iter().map(|x| x / norm).collect()
}
fn get_top_k_embeddings(
    query_embedding: &Vec<f32>,
    doc_embeddings: Vec<Vec<f32>>,
    doc_ids: Vec<usize>,
    similarity_top_k: usize,
) -> Vec<(usize, f32)> {
    let normalized_query = normalize(&query_embedding);

    let mut similarities: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .zip(doc_ids.iter())
        .map(|(doc_embedding, &doc_id)| {
            let normalized_doc = normalize(doc_embedding);
            let dot_product: f32 = normalized_query
                .iter()
                .zip(normalized_doc.iter())
                .map(|(q, d)| q * d)
                .sum();
            (doc_id, dot_product)
        })
        .collect();

    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    similarities.truncate(similarity_top_k);
    similarities
}

fn read_pdf(path: String) -> String{
    let bytes = std::fs::read(path).unwrap();
    let out = pdf_extract::extract_text_from_mem(&bytes).unwrap();
    out
}



async fn embed_ollama_data(model: &String, query: &str) -> Result<Vec<Vec<f32>>> {
    let ollama = Ollama::default();
    let request = GenerateEmbeddingsRequest::new(model.to_string(), query.into());
    let res = ollama.generate_embeddings(request).await.unwrap();
    Ok(res.embeddings)
}



fn parse_into_chunks(text: &str) -> Result<impl Iterator<Item = &str>>{
    let max_characters =Some(256).unwrap();
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let splitter  = TextSplitter::new(ChunkConfig::new(max_characters).with_sizer(tokenizer));
    let chunks = splitter.chunks(text);
    Ok(chunks.collect::<Vec<_>>().into_iter())

}

// fn get_tokenizer() -> Result<Tokenizer>{
//     let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
//     let bpe = bpe_builder
//         .dropout(0.1)
//         .unk_token("[UNK]".into())
//         .build()?;
//     let tokeni = Tokenizer::from_file(file_path).unwrap();
//     let mut tokenizer = Tokenizer::new(bpe);
//     Ok(tokenizer)
// }