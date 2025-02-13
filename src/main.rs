mod search;

#[tokio::main]
async fn main() {
    let test = "words words words";
    let mut vector_store =  search::VectorStore::new("deepseek-r1:32b");
    let filter = search::Filter::new("key".to_string(), "value".to_string());
    // vector_store.add(test, filter).await;
    let response = vector_store.search(test).await;
    println!("{:?}", response);
}
