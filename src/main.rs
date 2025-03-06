mod search;
use ollama_rs::generation::completion::request::GenerationRequest;
use ollama_rs::Ollama;



#[tokio::main]
async fn main() {
    let prompt = "how does trample work?";
    let mut vector_store =  search::VectorStore::new("deepseek-r1:32b");    
    let filter = search::Filter::new("key".to_string(), "value".to_string());
    vector_store.add_pdf("MagicCompRules 20250207.pdf".to_string(), filter).await;
    let response = vector_store.search(prompt).await;
    println!("{:?}", response.len());

    let base_prompt = "You are a Magic: The Gathering Judge.
    Explain how {} interact in a hypothetical scenario based on the following information:
    
    User Question: {}
    Card Details: {}
    Reasoning Requirements:
            Think about each card's abilities and effects separately.
            Address the specific user question using these abilities and rulings.
            If there is a colon, before the colon is a cost, after the colon is an effect.
            Explain how timing, priority, and state-based actions affect the interaction, if applicable.

    Be clear, concise, and ensure your explanation aligns with the rules of Magic: The Gathering. 
    Assume the user has a basic understanding of the game mechanics but may not grasp complex rulings.
    Do not reference cards outside of the question. Do not make examples that are outside of the given cards.
    Please only respond in plain text format
    Please only place your context within the <think> tags.
    ";
    let model = "deepseek-r1:32b".to_string();
    let sentences: Vec<&str> = response.iter().map(|node| node.sentence.as_str()).collect();
    let prompt_and_search = format!("{}, Question: {} Rules Context: {:?}", base_prompt, prompt, sentences);
    println!("{}", &prompt_and_search);
    let ollama = Ollama::default();
    let res = ollama.generate(GenerationRequest::new(model, prompt_and_search)).await;

    if let Ok(res) = res {
        println!("{}", res.response);
    }




}

// TODO: Add a test for the search function