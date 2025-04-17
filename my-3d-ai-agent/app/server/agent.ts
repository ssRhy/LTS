import dotenv from "dotenv";
dotenv.config();
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

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

// ==================== tool代码生成 的propmt ====================
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
       Focus exclusively on translating the description into working code. Include all requested elements while maintaining a clean, professional implementation.
`,
  ],
  ["human", "{input}"],
]);

const codeGenerationChain = codeGenerationPrompt.pipe(llm);

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
];
// ==================== 创建agent中介 ====================
const executionAgentPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `你是一个代码执行协调Agent，负责：
1. 调用验证工具（validate_threejs_code）检查代码完整性
2. 自动保存并通过send_code_to_websocket把代码发送到前端实时查看代码和代码所渲染的3d场景
3. 将问题反馈给生成Agent（executionAgentExecutor）`,
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
});

// ==================== 主流程 workflow？ ====================
async function mainAgent(userInput) {
  // 步骤1: 生成代码，调用codeGenerationChain
  const generationResult = await codeGenerationChain.invoke({
    input: userInput,
  });
  const generatedCode = generationResult.content;

  // 步骤2: 执行验证和预览，调用executionAgentExecutor
  const executionResult = await executionAgentExecutor.invoke({
    input: `验证并预览以下代码：\n${generatedCode}`,
    chat_history: [],
  });

  // 步骤3: 处理反馈
  if (executionResult.output.includes("验证通过")) {
    // 使用WebSocket发送代码
    await executionAgentExecutor.invoke({
      input: `通过WebSocket发送以下代码：\n${generatedCode}`,
      chat_history: [],
    });

    return {
      status: "success",
      code: generatedCode,
      preview: executionResult.output,
    };
  } else {
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
      } else {
        console.log("\n 需要修正：", result.feedback);
      }

      ask();
    });
  };

  ask();
}

// 启动系统
interactiveLoop();
