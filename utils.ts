import { Pinecone } from "@pinecone-database/pinecone";
import { pipeline } from "@xenova/transformers";
import { modelname, namespace, topK } from "./app/config";

let embedder: any = null;

export async function queryPineconeVectorStore(
  client: Pinecone,
  indexName: string,
  namespace: string,
  query: string
): Promise<string> {
  try {
    // Initialize the embedder if not already done
    if (!embedder) {
      embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }

    // Generate embeddings locally
    const output = await embedder(query, { pooling: 'mean', normalize: true });
    const queryEmbedding = Array.from(output.data);
    
    const index = client.Index(indexName);
    const queryResponse = await index.namespace(namespace).query({
      topK: 5,
      vector: queryEmbedding as any,
      includeMetadata: true,
      includeValues: false
    });

    if (queryResponse.matches.length > 0) {
      const concatenatedRetrievals = queryResponse.matches
        .map((match,index) =>`\nClinical Finding ${index+1}: \n ${match.metadata?.chunk}`)
        .join(". \n\n");
      return concatenatedRetrievals;
    } else {
      return "<nomatches>";
    }
  } catch (error: any) {
    console.error("Error in queryPineconeVectorStore:", error);
    throw new Error(`Failed to process query: ${error.message}`);
  }
}
