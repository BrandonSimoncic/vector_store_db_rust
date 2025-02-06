mod search;

#[tokio::main]
async fn main() {
    let test = "words words words";
    let mut vector_store =  search::VectorStore::new();
    vector_store.add(test).await;
}
