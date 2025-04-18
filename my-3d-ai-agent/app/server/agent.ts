import dotenv from "dotenv";
dotenv.config();
// 引入 RunnableSequence 和 StringOutputParser
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";

import { AzureChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { AgentExecutor, createToolCallingAgent } from "langchain/agents";

import { DynamicStructuredTool } from "@langchain/core/tools";

import readline from "readline";
import fs from "fs";
import path from "path";
import { z } from "zod";
import { WebSocketServer, WebSocket } from "ws";

// 创建WebSocket服务器
const wss = new WebSocketServer({
  port: process.env.WS_PORT ? parseInt(process.env.WS_PORT) : 3001,
});
let activeConnections: WebSocket[] = [];

// WebSocket服务器连接事件
wss.on("connection", (ws) => {
  console.log("Client connected to WebSocket");
  activeConnections.push(ws);

  // 设置心跳间隔
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
      console.log("Ping sent to client");
    }
  }, 30000); // 每30秒发送一次ping

  ws.on("close", () => {
    console.log("Client disconnected from WebSocket");
    clearInterval(pingInterval); // 清除心跳间隔
    activeConnections = activeConnections.filter((conn) => conn !== ws);
  });

  ws.on("message", (message) => {
    try {
      const data = JSON.parse(message.toString());
      console.log("Received message from client:", data);

      // 处理来自前端的消息
      if (data.type === "user_input" || data.type === "user_prompt") {
        console.log("User input:", data.content);
        // 调用mainAgent处理用户输入
        try {
          // 通知客户端正在处理请求
          ws.send(
            JSON.stringify({
              type: "agent_thinking",
              content: "正在生成您的Three.js代码...",
            })
          );

          // 发送generation_started消息
          ws.send(
            JSON.stringify({
              type: "generation_started",
            })
          );

          mainAgent(data.content).then((result) => {
            // 发送处理结果状态消息
            ws.send(
              JSON.stringify({
                type: "agent_message",
                content: `已处理请求: ${
                  result.status === "success" ? "成功" : "需要修正"
                }`,
              })
            );

            // 如果代码生成成功，确保代码被发送（虽然send_code_to_websocket工具也会发送）
            if (result.status === "success" && result.code) {
              // 发送generation_completed消息
              ws.send(
                JSON.stringify({
                  type: "generation_completed",
                  code: result.code,
                })
              );
            } else if (result.status === "needs_revision") {
              // 发送错误反馈信息
              ws.send(
                JSON.stringify({
                  type: "error",
                  content: result.feedback,
                })
              );
            }
          });
        } catch (error) {
          console.error("Error processing request:", error);
          ws.send(
            JSON.stringify({
              type: "error",
              content: "处理请求时发生错误，请重试",
            })
          );
        }
      }
    } catch (error) {
      console.error("Error parsing message:", error);
    }
  });
});

// 配置 Azure OpenAI
const llm = new AzureChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
  azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
  azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
  azureOpenAIApiVersion: "2024-02-15-preview",
});

// ==================== 初始化 Embeddings 和 Vector Store ====================
// 配置 Azure OpenAI Embeddings
const embeddings = new OpenAIEmbeddings({
  azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
  azureOpenAIApiDeploymentName:
    process.env.AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME ||
    "text-embedding-ada-002",
  azureOpenAIApiVersion: "2024-02-15-preview",
});

// 创建向量存储（初始为空，将在初始化时填充）
let vectorStore;

// 初始化代码存储
async function initializeCodeMemory() {
  // 检查是否有历史代码存储文件
  const historyPath = path.join(process.cwd(), "code_history.json");
  if (fs.existsSync(historyPath)) {
    try {
      const historyData = JSON.parse(fs.readFileSync(historyPath, "utf-8"));

      // 将历史代码转换为Document对象数组
      const docs = historyData.map(
        (entry) =>
          new Document({
            pageContent: entry.code,
            metadata: {
              prompt: entry.prompt,
              timestamp: entry.timestamp,
              id: entry.id,
            },
          })
      );

      // 使用fromDocuments创建向量存储
      vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
      console.log(`加载了 ${historyData.length} 个历史代码示例到向量存储`);
    } catch (error) {
      console.error("加载历史代码时出错:", error);
    }
  } else {
    console.log("没有找到历史代码存储，创建新的向量存储");
    // 创建空的向量存储
    vectorStore = new MemoryVectorStore(embeddings);
  }
}

// 保存新代码到历史记录
async function saveCodeToHistory(prompt, code) {
  const historyPath = path.join(process.cwd(), "code_history.json");
  let historyData = [];

  // 读取现有历史
  if (fs.existsSync(historyPath)) {
    try {
      historyData = JSON.parse(fs.readFileSync(historyPath, "utf-8"));
    } catch (error) {
      console.error("读取历史文件时出错:", error);
    }
  }

  // 添加新条目
  const newEntry = {
    id: Date.now().toString(),
    prompt: prompt,
    code: code,
    timestamp: new Date().toISOString(),
  };

  historyData.push(newEntry);

  // 保存回文件
  fs.writeFileSync(historyPath, JSON.stringify(historyData, null, 2), "utf-8");

  // 同时添加到向量存储
  await vectorStore.addDocuments([
    new Document({
      pageContent: code,
      metadata: {
        prompt: prompt,
        timestamp: newEntry.timestamp,
        id: newEntry.id,
      },
    }),
  ]);

  console.log("代码已保存到历史记录");
  return newEntry.id;
}

// 从向量存储中检索相关代码
async function retrieveSimilarCode(prompt, k = 3) {
  // 创建一个检索器，指定返回k个最相似的文档
  const retriever = vectorStore.asRetriever(k);
  // 使用检索器检索相关代码
  const results = await retriever.invoke(prompt);
  return results;
}

// ==================== 对话记忆与流程优化 ====================
// 创建对话记忆
const memory = new BufferMemory({
  returnMessages: true,
  memoryKey: "chat_history",
  inputKey: "input",
  outputKey: "output",
});

// 创建对话链用于处理用户输入，包含记忆功能
const conversationChain = new ConversationChain({
  llm: llm,
  memory: memory,
  prompt: ChatPromptTemplate.fromMessages([
    ["system", "你是Three.js代码生成助手。分析用户需求，提供有用的建议。"],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]),
  verbose: true,
});

// ==================== tool代码生成 的prompt ====================
const codeGenerationPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `你是一名专业的Three.js代码生成专家，严格遵循以下规则：
  CODE STRUCTURE:
      - 将用户输入的自然语言详细描述解析，然后再生成代码
      - Complete HTML document with proper head/body sections
      - ES Module imports using Three.js version 0.153.0
      - Scene/camera/renderer initialization
      - User-requested objects and features
      - Animation loop and event handlers
      - 单HTML文件实现，便于在线平台直接运行
      - 内联CSS样式
   TECHNICAL REQUIREMENTS:
      - Modern ES Module imports with importmap
      - ES Module Shims for compatibility
      - Proper component initialization sequence
      - Responsive design with resize handlers
      - Clean, commented code structure
      
   重要：我会提供两种类型的上下文信息：
   1. 相似代码示例：这些是之前生成的与当前需求相似的代码，可以参考其结构和实现方式
   2. 对话历史：这些是之前与用户的交互，可以从中了解用户的偏好和额外的上下文信息
   
   请根据提供的上下文信息，结合用户的新需求，生成最适合的Three.js代码。
`,
  ],
  new MessagesPlaceholder("chat_history"),
  [
    "human",
    "以下是与当前需求相似的代码示例:\n{similar_code}\n\n用户需求: {input}",
  ],
]);

// ==================== 使用RunnableSequence和StringOutputParser ====================
// 创建用于获取上下文信息的函数
async function getSimilarCode(input) {
  const similarCodeDocs = await retrieveSimilarCode(input);
  return similarCodeDocs.length > 0
    ? similarCodeDocs
        .map(
          (doc) =>
            `示例 (来自: ${doc.metadata.prompt}):\n${doc.pageContent.substring(
              0,
              1500
            )}...\n\n`
        )
        .join("\n")
    : "没有找到相似代码示例。";
}

async function getChatHistory() {
  const memoryVariables = await memory.loadMemoryVariables({});
  return memoryVariables.chat_history || [];
}

// 创建代码生成序列
const codeGenerationSequence = RunnableSequence.from([
  {
    // 输入映射函数，将用户输入转换为提示模板所需的输入格式
    async formatter(input) {
      if (typeof input === "string") {
        input = { input };
      }

      // 获取聊天历史
      const chatHistory = await getChatHistory();

      // 获取相似代码
      const similarCode = await getSimilarCode(input.input);

      // 返回格式化后的输入对象
      return {
        input: input.input,
        chat_history: chatHistory,
        similar_code: similarCode,
      };
    },
  },
  // 应用提示模板
  codeGenerationPrompt,
  // 使用LLM生成响应
  llm,
  // 使用StringOutputParser解析输出为字符串
  new StringOutputParser(),
]);

// ==================== 代码传递 tool ====================
const tools = [
  new DynamicStructuredTool({
    name: "send_code_to_websocket",
    description: "将生成的Three.js代码保存并通过websocket实时发送到前端",
    schema: z.object({
      code: z.string().describe("需要发送的代码内容"),
    }),
    func: async ({ code }) => {
      // 保存代码到文件
      const outputPath = path.join(process.cwd(), "output.html");
      fs.writeFileSync(outputPath, code, "utf-8");
      console.log(`代码已保存到: ${outputPath}`);

      // 发送代码到所有连接的WebSocket客户端
      const connectedClients = activeConnections.length;
      activeConnections.forEach((client) => {
        if (client.readyState === 1) {
          // WebSocket.OPEN
          client.send(
            JSON.stringify({
              type: "code_generated",
              code: code,
              requestId: Date.now().toString(),
            })
          );
        }
      });

      return `代码已通过WebSocket发送到${connectedClients}个前端客户端，并保存到${outputPath}`;
    },
  }),
  // ==================== 代码检查 tool ====================
  new DynamicStructuredTool({
    name: "validate_threejs_code",
    description: `验证Three.js代码完整性。此工具将检查以下关键元素： 
  - THREE.Scene：用于创建3D场景。
  - PerspectiveCamera：透视摄像机，用于控制视角。
  - WebGLRenderer：渲染器，负责将场景渲染到屏幕上。
  - animate()：动画循环函数，确保场景不断更新。
  - requestAnimationFrame：用于实现平滑的动画渲染。
   - 使用ES Module Shims确保兼容性
   - 避免使用已弃用的build/three.js和build/three.min.js
   - 将script标签改为type="module"
   - 使用ES模块方式导入Three.js和OrbitControls
   - 添加了ES Module Shims库，提高浏览器兼容性`,
    schema: z.object({
      code: z.string().describe("需要验证的代码内容"),
    }),
    func: async ({ code }) => {
      const requiredElements = [
        "THREE.Scene",
        "PerspectiveCamera",
        "WebGLRenderer",
        "animate()",
        "requestAnimationFrame",
      ];

      const missing = requiredElements.filter((el) => !code.includes(el));
      return missing.length > 0
        ? `缺少关键元素: ${missing.join(", ")}`
        : "代码验证通过";
    },
  }),
  // ==================== 代码记忆检索 tool ====================
  new DynamicStructuredTool({
    name: "retrieve_similar_code",
    description: "根据用户的描述检索相似的代码示例",
    schema: z.object({
      query: z.string().describe("用户的代码需求描述"),
      limit: z.number().optional().describe("要检索的代码示例数量，默认为3"),
    }),
    func: async ({ query, limit = 3 }) => {
      const results = await retrieveSimilarCode(query, limit);
      if (results.length === 0) {
        return "没有找到相似代码。";
      }

      return results
        .map(
          (doc, index) =>
            `示例 ${index + 1} (来自: ${
              doc.metadata.prompt
            }):\n${doc.pageContent.substring(0, 500)}...\n`
        )
        .join("\n\n");
    },
  }),
];

// ==================== 创建agent中介 ====================
const executionAgentPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `你是一个代码执行协调Agent，负责：
1. 调用验证工具（validate_threejs_code）检查代码完整性
2. 自动保存并通过send_code_to_websocket把代码发送到前端实时查看代码和代码所渲染的3d场景
3. 将问题反馈给生成Agent（executionAgentExecutor）
4. 使用retrieve_similar_code工具检索相似代码示例作为参考`,
  ],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

// 调用 AgentExecutor等工具
const executionAgent = createToolCallingAgent({
  llm,
  tools,
  prompt: executionAgentPrompt,
});

// ==================== 协调多个工具 和 执行任务 ===================
const executionAgentExecutor = new AgentExecutor({
  agent: executionAgent,
  tools,
  verbose: true,
  // 不使用memory避免冲突
});

// ==================== 主流程 workflow ====================
// 定义GenerationContext接口
interface GenerationContext {
  chatHistory: any[];
  similarCode: string;
}

async function mainAgent(userInput) {
  // 步骤0: 临时跳过分析用户输入
  const analysisResult = {
    needsConversationMemory: true,
    needsCodeMemory: false, // 设置为false禁用代码记忆
  };

  console.log("临时设置: 仅使用对话记忆，跳过代码记忆");

  // 步骤1: 生成代码前准备上下文
  let generationContext: GenerationContext = {
    chatHistory: [],
    similarCode: "跳过代码记忆功能，使用简单模板。",
  };

  // 获取对话记忆
  // 从记忆中获取历史对话
  const memoryVariables = await memory.loadMemoryVariables({});
  generationContext.chatHistory = memoryVariables.chat_history || [];
  console.log("已加载对话记忆");

  // 临时跳过代码记忆检索，使用固定内容
  generationContext.similarCode = "跳过向量搜索功能验证。";

  // 步骤2: 生成代码，调用增强的codeGenerationChain
  console.log("开始生成代码...");
  const generationResult = await codeGenerationPrompt.pipe(llm).invoke({
    input: userInput,
    similar_code: generationContext.similarCode,
    chat_history: generationContext.chatHistory || [],
  });

  const generatedCode = generationResult.content;
  console.log("代码生成完成，长度:", generatedCode.length);

  // 步骤3: 执行验证和预览，调用executionAgentExecutor
  const executionResult = await executionAgentExecutor.invoke({
    input: `验证并预览以下代码：\n${generatedCode}`,
  });

  // 添加测试日志
  console.log("记忆测试 - 当前记忆内容:");
  const currentMemory = await memory.loadMemoryVariables({});
  console.log(JSON.stringify(currentMemory, null, 2));

  // 步骤4: 处理反馈并保存到记忆
  if (executionResult.output.includes("验证通过")) {
    // 临时跳过向量存储
    // await saveCodeToHistory(userInput, generatedCode);

    // 记录这次成功的对话到对话记忆
    await memory.saveContext(
      { input: userInput },
      { output: "代码生成成功，已通过验证" }
    );

    console.log("记忆测试 - 保存成功记录后:");
    const updatedMemory = await memory.loadMemoryVariables({});
    console.log(JSON.stringify(updatedMemory, null, 2));

    // 使用WebSocket发送代码
    await executionAgentExecutor.invoke({
      input: `通过WebSocket发送以下代码：\n${generatedCode}`,
    });

    return {
      status: "success",
      code: generatedCode,
      preview: executionResult.output,
    };
  } else {
    // 记录这次失败的对话到对话记忆
    await memory.saveContext(
      { input: userInput },
      { output: `代码生成需要修正: ${executionResult.output}` }
    );

    console.log("记忆测试 - 保存失败记录后:");
    const updatedMemory = await memory.loadMemoryVariables({});
    console.log(JSON.stringify(updatedMemory, null, 2));

    return {
      status: "needs_revision",
      feedback: executionResult.output,
      original_code: generatedCode,
    };
  }
}

// ==================== 交互界面 ====================
async function interactiveLoop() {
  console.log("Three.js 智能生成系统正在运行...");
  console.log("WebSocket服务器监听端口:", process.env.WS_PORT || 3001);

  // 临时跳过初始化代码记忆
  console.log("跳过向量存储初始化...");
  // await initializeCodeMemory();

  // 测试记忆功能
  console.log("测试记忆功能...");
  await memory.saveContext({ input: "测试输入" }, { output: "测试记忆初始化" });
  const testMemory = await memory.loadMemoryVariables({});
  console.log("初始记忆测试:", JSON.stringify(testMemory, null, 2));

  // 保留命令行接口作为备用
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const ask = () => {
    rl.question("\n请输入物体描述（或输入exit退出）：", async (input) => {
      if (input.toLowerCase() === "exit") {
        rl.close();
        return;
      }

      const result = await mainAgent(input);

      if (result.status === "success") {
        console.log("\n代码生成成功，已通过WebSocket发送!");
      } else {
        console.log("\n需要修正：", result.feedback);
      }

      ask();
    });
  };

  ask();
}

// 启动系统
interactiveLoop();

