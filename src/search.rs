use std::fs::Metadata;
use std::io::Error;

use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::Tokenizer;

struct Filter {
    key: String,
    value: String,
}
struct Node {
    embedding: Vec<Vec<f32>>,
    sentence: &str,
    text_id: usize,
    metadata: Filter,
}
impl Node {
    async fn new (words: &str, id: usize, metadat: Filter)-> Node{
        let embeddings = embed_ollama_data("llama3.2:latest", &words).await.unwrap();
        Node {
            embedding: embeddings,
            sentence:words,
            text_id: id,
            metadata: metadat,
        }
    }
    
}

pub struct VectorStore {
    nodes: Vec<Node>
}

impl VectorStore {
    pub fn new() -> Self {
        VectorStore {
            nodes: Vec::new(),
        }
    }
    //Get Embedding
    pub fn get(&self, text_id: usize) -> Option<&Vec<Vec<f32>>> {
        self.nodes.get(text_id).map(|node| &node.embedding)
    }
    //Add nodes to index
    pub async fn add(&mut self, words: &str, metadata: Filter){
        let id = self.nodes.len() + 1;
        let node = Node::new(words, id, metadata).await;
        self.nodes.push(node);
    }
    //Delete nodes using ref_doc_id
    pub fn delete(&mut self, ref_doc_id: usize){
        let item = self.nodes.iter().position(|r| r.text_id == ref_doc_id).unwrap();
        self.nodes.remove(item);
    }
    //Get nodes for response
    pub fn query(&self, query: &Node, filters: Vec<Filter>) -> Vec<Node>{


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
            let similarities_and_node_ids = get_top_k_embeddings(
                &query.embedding, 
                doc_embeddings, 
                doc_ids,
            5);
            result_nodes = similarities_and_node_ids
            .iter()
            .filter_map(|(id, _)| self.nodes.iter().find(|node| node.text_id == *id))
            .collect();
        }
        
        

    result_nodes
    }

    //Persistent SimpleFectorStore to a dir
    fn persist(){}


}

fn filter_nodes(nodes: &VectorStore, filters: Vec<Filter>) -> Vec<Node> {
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
                filtered_nodes.push(*node.clone());
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


use ollama_rs::generation::embeddings::request::GenerateEmbeddingsRequest;
use ollama_rs::Ollama;

async fn embed_ollama_data(model: &str, query: &str) -> Result<Vec<Vec<f32>>, Error> {
    let ollama = Ollama::default();

    let request = GenerateEmbeddingsRequest::new(model.to_string(), query.into());
    let res = ollama.generate_embeddings(request).await.unwrap();
    Ok(res.embeddings)
}



fn parse_into_chunks(text: &str) -> &(impl Iterator<Item = &str>){
    let max_characters =Some(256);
    let tokenizer = Tokenizer::from_pretrained("PreTrainedTokenizerFast", None).unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(max_characters).with_sizer(tokenizer));
    let chunks = &splitter.chunks(text);
    chunks
}