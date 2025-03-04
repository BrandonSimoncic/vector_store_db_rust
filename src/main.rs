mod search;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;



#[tokio::main]
async fn main() {
    let prompt = "how does trample work?";
    let mut vector_store =  search::VectorStore::new("deepseek-r1:32b");    
    let filter = search::Filter::new("key".to_string(), "value".to_string());
    vector_store.add_pdf("MagicCompRules 20250207.pdf".to_string(), filter).await;
    let response = vector_store.query(prompt, filter);
    println!("{:?}", response.len());


    let model = "deepseek-r1:32b".to_string();
    let prompt_and_search = format!("{} {:?}", prompt, response);
    let ollama = Ollama::default();
    let res = ollama.generate(GenerationRequest::new(model, prompt_and_search)).await;

    if let Ok(res) = res {
        println!("{}", res.response);
    }




}
