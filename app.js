/******************************************************************************
 * 1. IMPORTS & SETUP
 ******************************************************************************/
require("dotenv").config();
const fs = require("fs");
const path = require("path");
const express = require("express");
const helmet = require("helmet");
const http = require("http");
const { Server } = require("socket.io");
const bodyParser = require("body-parser");
const { Configuration, OpenAIApi } = require("openai");

const app3 = express();
const server = http.createServer(app3);
const io = new Server(server);




/******************************************************************************
 * 2. HELMET CONTENT SECURITY POLICY => please remove for proper functioning on different environments. *! !!
 ******************************************************************************/
app3.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        "connect-src": [
          "https://chat.lovethecode.net",
          "https://app3.lovethecode.net",
          "https://app2.lovethecode.net",
          "https://app.lovethecode.net",
          "wss://app.lovethecode.net",
          "https://ang.lovethecode.net",
        ],
        "frame-ancestors": [
          "'self'",
          "https://chat.lovethecode.net",
          "https://app.lovethecode.net",
          "https://app2.lovethecode.net",
          "https://app3.lovethecode.net",
          "wss://app.lovethecode.net",
          "https://ajax.googleapis.com",
          "https://ang.lovethecode.net",
        ],
        "script-src-elem": [
          "'unsafe-inline'",
          "https://chat.lovethecode.net",
          "https://cdn.socket.io",
          "https://app.lovethecode.net",
          "https://app2.lovethecode.net",
          "https://app3.lovethecode.net",
          "wss://app.lovethecode.net",
          "https://ajax.googleapis.com",
          "https://ang.lovethecode.net",
        ],
        "frame-src": [
          "'self'",
          "https://chat.lovethecode.net",
          "https://app.lovethecode.net",
          "https://app2.lovethecode.net",
          "https://app3.lovethecode.net",
          "wss://app.lovethecode.net",
          "https://ajax.googleapis.com",
          "https://ang.lovethecode.net",
        ],
      },
    },
  })
);

app3.use(bodyParser.json());
app3.use(bodyParser.urlencoded({ extended: false }));

/******************************************************************************
 * 3. OPENAI CONFIG
 ******************************************************************************/
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const EMBEDDINGS_MODEL = "text-embedding-3-small"; // "text-embedding-ada-002";
const CHAT_MODEL = "gpt-4o-mini"; // or "gpt-3.5-turbo"

/******************************************************************************
 * 4. GLOBAL VECTOR STORE (for all users)
 ******************************************************************************/
let vectorStore = [];

/******************************************************************************
 * 5. PER-USER CONVERSATION OBJECT
 ******************************************************************************/
let userConversations = {}; 
// e.g. userConversations[socketId] = [ { role: "user"|"assistant", content: "..." }, ... ]

/******************************************************************************
 * 6. LOAD & EMBED MEETINGS.JSON
 ******************************************************************************/
function loadJsonFile(filePath) {
  const raw = fs.readFileSync(filePath, "utf8");
  return JSON.parse(raw);
}

/** linkify function */
function linkify(text) {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  return text.replace(urlRegex, (url) => {
    return `<a href="${url}" target="_blank">${url}</a>`;
  });
}

// Cosine similarity
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function createEmbeddingForText(text) {
  const response = await openai.createEmbedding({
    model: EMBEDDINGS_MODEL,
    input: text,
  });
  return response.data.data[0].embedding;
}

async function buildVectorStore(filePath) {
  const data = loadJsonFile(filePath);
  console.log(`Loaded ${data.length} items from ${filePath}.`);

  for (const record of data) {
    const textBlock = `
Name: ${record.name}
Location: ${record.location}
Day: ${record.day}
Time: ${record.time}
Region: ${record.region}
Type: ${record.type}
URL: ${record.url || "No URL"}
    `.trim();

    try {
      const embedding = await createEmbeddingForText(textBlock);
      vectorStore.push({ embedding, text: textBlock });
    } catch (err) {
      console.error("Error embedding record:", err.message);
    }
  }
  console.log(`Vector store built with ${vectorStore.length} items.`);
}

async function topNMatches(query, n = 8) {
  const queryEmbedding = await createEmbeddingForText(query);

  const scored = vectorStore.map((item) => {
    const sim = cosineSimilarity(queryEmbedding, item.embedding);
    return { text: item.text, similarity: sim };
  });

  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, n);
}

/******************************************************************************
 * 7. CREATE A SINGLE RESPONSE (BUT RETAIN HISTORY PER USER)
 ******************************************************************************/
async function answerUserQuery(socketId, userQuery) {
  // 1) Retrieve the conversation for this user
  let conversation = userConversations[socketId] || [];
  
  // 2) We do a retrieval for the new query
  const matches = await topNMatches(userQuery, 8);
  const contextSnippet = matches
    .map(
      (m, idx) => `Match #${idx + 1} (sim=${m.similarity.toFixed(8)}):\n${m.text}`
    )
    .join("\n---\n");

  // 3) We'll inject a system message referencing the snippet
  const systemMessage = {
    role: "system",
    content:
      "You are a Meeting assistant. Render all output as HTML. Outline all output in nested bulleted lists. Treat all textural information as case insensitve, and treat all information lower case, but output normally. treat the word AAC as if it were the Atlantic Alano Club as a meeting location. The word MWA as a meeting location should be output M.W.A.  You answer questions about meetings in a region given in the region field. You answer questions about meetings at certain times. You will output the time field in the am/pm method, not in military time. You answer questions about meetings on specific days given by the day field. (0 is Sunday, 1 is Monday, 2 is Tuesday, 3 is Wednesday, 4 is Thursday, 5 is Friday, 6 is Saturday). You answer questions about meetings at various places given by the location field. the meeting `url` field should be ignored in all cases. We will be focusing on the `conference_url` field, instead. Hybrid meetings count as both an in_person meeting and as an online meeting. Hybrid meetings are located at the location field parameter. You will provide details on Meetings, such as name, location, address, and link of `conference_url`. If the `attendance_option` is hybrid or online, you will output the corresponding `conference_url` field (also referred to as conference URL) to the meeting online room with zoom, google meet or other vendor. Do not output a url if it is in_person, output a google map url with meeting location adresses formatted within the url. Format all URLs as hyperlinks. Render all html markup. When referencing something on the website either write a hypertext with the exact link or when describing sections of the website or page provide some markers to help navigate. Format all URLs as hyperlinks. Render all html markup. You have access to the following relevant text from 'meetings.json':\n\n" +
      contextSnippet +
      "\n\nUse it to answer the user's query. Continue the conversation context as needed.",
  };

  // 4) Append the user’s new message to their conversation
  conversation.push({ role: "user", content: `${userQuery} + Please include up to 12 meeting listings in your response, if the user is requesting meetings in their query.` });

  // 5) Combine all messages with the new system message at the front
  //    Typically, you put the system messages at the start. 
  //    Then the user & assistant messages follow.
  const finalMessages = [systemMessage, ...conversation];

  // 6) Call GPT for a single new response
  const completion = await openai.createChatCompletion({
    model: CHAT_MODEL,
    messages: finalMessages,
    temperature: 0.7,
  });

  const aiAnswer = completion.data.choices[0].message.content;

  // 7) Store the assistant's response into the user's conversation
  conversation.push({ role: "assistant", content: aiAnswer });

  // 8) Update the userConversations map 
  userConversations[socketId] = conversation; // store updated array

  // 9) Return the single new answer
  return aiAnswer;
}

/******************************************************************************
 * 8. EXPRESS ROUTE (OPTIONAL) 
 ******************************************************************************/
app3.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "views/index.html"));
});
app3.use(express.static('public'));
app3.use(express.static(path.join("node_modules/dist")));
/******************************************************************************
 * 9. SOCKET.IO - ONE CONVERSATION PER USER
 ******************************************************************************/
io.on("connection", (socket) => {
  console.log("A user connected:", socket.id);

  // Initialize an empty conversation for this user
  userConversations[socket.id] = [];

  // Listen for user queries
  socket.on("chat message", async (userQuery) => {
    console.log(`User ${socket.id} asked:`, userQuery);

    try {
      // Make exactly one GPT call
      const aiAnswer = await answerUserQuery(socket.id, userQuery);
      
      // Emit exactly one response to this user 
      // (Or broadcast to all if you prefer, but typically you'd use socket.emit)
      socket.emit("chat response", aiAnswer);

      console.log(`Responded to ${socket.id} with:`, aiAnswer);
    } catch (err) {
      console.error("Error answering query:", err.message);
      socket.emit("chat response", "Sorry, an error occurred while processing your request.");
    }
  });

  // If user disconnects, clean up their conversation memory
  socket.on("disconnect", () => {
    console.log(`User ${socket.id} disconnected.`);
    delete userConversations[socket.id];
  });
});

/******************************************************************************
 * 10. START THE SERVER
 ******************************************************************************/
const PORT = process.env.PORT || 3000;
server.listen(PORT, async () => {
  console.log("Listening on port:", PORT);
  try {
    await buildVectorStore("meetings.json");
    console.log("Vector store ready. Each user now has separate GPT context!");
  } catch (err) {
    console.error("Error building vector store:", err.message);
  }
});
