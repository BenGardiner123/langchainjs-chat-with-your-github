import { OpenAI } from "langchain/llms/openai";
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { Document } from "langchain/document";
import * as fs from "fs";
import * as dotenv from "dotenv";
dotenv.config();

function normalizeDocuments(docs: Document<Record<string, any>>[]) {
  return docs.map((doc) => {
    if (typeof doc.pageContent === "string") {
      return doc.pageContent;
    } else if (Array.isArray(doc.pageContent)) {
      return (doc.pageContent as string[]).join("\n");
    }
    throw new Error("Invalid pageContent type");
  });
}

const model = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

async function loadDocuments() {
  console.log("Loading docs");
  const loader = new GithubRepoLoader(
    "https://github.com/BenGardiner123/langchainjs-typescript",
    {
      branch: "main",
      recursive: true,
      unknown: "warn",
      accessToken: process.env.GITHUB_ACCESS_TOKEN,
    }
  );
  const docs = await loader.load();
  console.log("Docs loaded", docs);
  return docs;
}

export const run = async () => {
  const directory = "./vectorstore";

  try {
    // Check if the files exist in the directory
    const argsFileExists = fs.existsSync(`${directory}/args.json`);
    const docstoreFileExists = fs.existsSync(`${directory}/docstore.json`);

    // To save money and time, we only want to load the documents and create the vector store if we need to
    if (!argsFileExists || !docstoreFileExists) {
      // At least one of the files doesn't exist in the directory
      // Load documents, create vector store, and save them

      const docs = await loadDocuments();

      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
      });

      const normalizedDocs = normalizeDocuments(docs);

      const splitDocs = await textSplitter.createDocuments(normalizedDocs);

      // Create a vector store for the documents using HNSWLib
      const vectorStore = await HNSWLib.fromDocuments(
        splitDocs,
        new OpenAIEmbeddings()
      );

      // Save the vector store to the directory
      await vectorStore.save(directory);
    }

    // Load the vector store from the directory
    const loadedVectorStore = await HNSWLib.load(
      directory,
      new OpenAIEmbeddings()
    );
    // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
    const chain = RetrievalQAChain.fromLLM(
      model,
      loadedVectorStore.asRetriever()
    );
    const res = await chain.call({
      query: `What can you tell me about the repository ? be specific Can you see a folder called "src". If you can see a folder called "src" can you tell me the name of the files inside it?`,
    });
    console.log({ res });

    // const followUp = await chain.call({
    //   query: `Can you see a folder called "src". If you can see a folder called "src" can you tell me the name of the files inside it?`,
    //   context: res.context,
    // });
    // console.log({ followUp });
  } catch (error) {
    console.error("An error occurred:", error);
  }
};

run();
