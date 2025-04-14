import { AzureChatOpenAI } from "@langchain/openai";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { BufferMemory } from "langchain/memory";
import * as readline from "readline";
import * as dotenv from "dotenv";
import { WebSocketServer, WebSocket } from "ws";
import type { IncomingMessage } from "http";
import * as http from "http";
import * as path from "path";
import 'dotenv/config';
import Module from "module";
import { create } from "domain";


// Type definitions
type ClientMessage = {
  type: string;
  content?: string;
  code?: string;
  requestId?: string;
  isValid?: boolean;
  message?: string;
  description?: string;
  status?: string;
  stage?: string;
  result?: string;
  error?: string;
};

// WebSocket with additional properties for our application
type ExtendedWebSocket = WebSocket & {
  isAlive: boolean;
};

// Get current file directory path
const __filename = new URL(import.meta.url).pathname;
const __dirname = path.dirname(__filename);

// Define project constants
const LANGCHAIN_PROJECT = "threejs-langchain-agent";

// Load environment variables from .env file
dotenv.config({ path: path.resolve(__dirname, "./.env") });
// Fallback to root .env if server .env doesn't have all required variables
dotenv.config({ path: path.resolve(__dirname, "../../.env") });

// Debug environment variables
console.log("Environment variables check:");
console.log("AZURE_OPENAI_API_KEY exists:", !!process.env.AZURE_OPENAI_API_KEY);
console.log("AZURE_OPENAI_ENDPOINT exists:", !!process.env.AZURE_OPENAI_ENDPOINT);
console.log("AZURE_OPENAI_API_VERSION exists:", !!process.env.AZURE_OPENAI_API_VERSION);
console.log("AZURE_OPENAI_API_DEPLOYMENT_NAME exists:", !!process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME);

// Configure WebSocket server
const PORT = process.env.WS_PORT || 3001;
console.log(`WebSocket server will use port: ${PORT}`);
const server: http.Server = http.createServer();
const wss = new WebSocketServer({
  server,
  // Add heartbeat detection to maintain stable connections
  clientTracking: true,
  perMessageDeflate: {
    zlibDeflateOptions: {
      chunkSize: 1024,
      memLevel: 7,
      level: 3,
    },
    zlibInflateOptions: {
      chunkSize: 10 * 1024,
    },
    concurrencyLimit: 10,
    threshold: 1024,
  },
});

// Store active connections
let activeConnections = new Set<ExtendedWebSocket>();

// Handle server errors
server.on("error", (error) => {
  console.error("HTTP server error:", error);
  // Try to restart the server
  setTimeout(() => {
    try {
      server.close();
      server.listen(PORT);
      console.log(`Server has attempted to restart, listening on port ${PORT}`);
    } catch (e) {
      console.error("Server restart failed:", e);
    }
  }, 5000);
});

// WebSocket server setup
wss.on("connection", (ws: WebSocket, req: IncomingMessage) => {
  const extendedWs = ws as ExtendedWebSocket;
  const ip = req.socket.remoteAddress;
  console.log(`Client connected [${ip}]`);
  activeConnections.add(extendedWs);

  // Set up heartbeat detection
  extendedWs.isAlive = true;
  extendedWs.on("pong", () => {
    extendedWs.isAlive = true;
  });

  extendedWs.on("message",(message: Buffer | ArrayBuffer | Buffer[]) => {
    try {
      const data = JSON.parse(message.toString());
      console.log("Message received:", data);

      // Handle based on message type
      if (data.type === "user_prompt" || data.type === "user_input") {
        handleUserPrompt(data.content, extendedWs);
      } else if (data.type === "tool_response") {
        // Handle tool response
        broadcastMessage(data);
      } else {
        // Handle other message types
        console.log(`Unhandled message type: ${data.type}`);
      }
    } catch (error) {
      console.error("Error processing message:", error);
      sendErrorToClient(extendedWs, "Unable to process message");
    }
  });

  extendedWs.on("close", () => {
    console.log("Client disconnected");
    activeConnections.delete(extendedWs);
  });

  extendedWs.on("error", (error) => {
    console.error("WebSocket error:", error);
    activeConnections.delete(extendedWs);
  });

  // Send connection success message
  extendedWs.send(
    JSON.stringify({
      type: "connection_status",
      status: "connected",
      message: "WebSocket connection established",
    })
  );
});

// Heartbeat detection interval
const heartbeatInterval = setInterval(() => {
  wss.clients.forEach((ws: WebSocket) => {
    const extendedWs = ws as ExtendedWebSocket;
    if (extendedWs.isAlive === false) {
      console.log("Detected disconnected connection, closing...");
      activeConnections.delete(extendedWs);
      return extendedWs.close();
    }

    extendedWs.isAlive = false;
    extendedWs.ping(() => {});
  });
}, 30000);

// Clean up when server closes
wss.on("close", () => {
  clearInterval(heartbeatInterval);
});

// Message handling functions
const handleUserPrompt = async (
  prompt: string,
  ws: ExtendedWebSocket
): Promise<void> => {
  try {
    // Send thinking message
    sendToClient(ws, {
      type: "ai_thinking",
      message: "Thinking about how to create a 3D scene...",
    });

    // Send workflow start message
    sendToClient(ws, {
      type: "workflow_started",
      stage: "analysis",
      message: "Starting to analyze user request",
    });

    const workflow = createConversationalWorkflow(ws);
    const result = await workflow(prompt);

    // Send workflow completion message
    sendToClient(ws, {
      type: "workflow_completed",
      result: "success",
    });

    // Compatible with older clients
    sendToClient(ws, {
      type: "agent_complete",
    });
  } catch (error) {
    const err = error as Error;
    console.error("Error processing user input:", err);
    sendErrorToClient(ws, `Error processing request: ${err.message}`);

    // Send workflow completion (failure) message
    sendToClient(ws, {
      type: "workflow_completed",
      result: "failed",
      error: err.message,
    });
  }
};

// Send error message to client
const sendErrorToClient = (ws: ExtendedWebSocket, message: string): void => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(
      JSON.stringify({
        type: "error",
        message,
      })
    );
  }
};

// Send message to client
const sendToClient = (ws: ExtendedWebSocket, data: ClientMessage): void => {
  if (ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify(data));
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  }
};

// Broadcast message to all connections
const broadcastMessage = (data: ClientMessage): void => {
  activeConnections.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      try {
        client.send(JSON.stringify(data));
      } catch (error) {
        console.error("Failed to broadcast message:", error);
      }
    }
  });
};

// Configure Azure OpenAI
const createAzureLLM = new AzureChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
  azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
  azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
  azureOpenAIApiVersion: "2024-02-15-preview",
});

// Tool definition layer - modified to use WebSocket communication
const createBasicTools = (ws: ExtendedWebSocket): DynamicStructuredTool[] => [
  new DynamicStructuredTool({
    name: "code_validator",
    description: "Validate Three.js code completeness",
    schema: z.object({
      code: z.string().describe("Complete code to validate"),
    }),
    func: async ({ code }: { code: string }): Promise<string> => {
      const isValid = code.includes("THREE") && code.includes("new Scene()");

      // Send validation result via WebSocket
      sendToClient(ws, {
        type: "validation_result",
        isValid,
        message: isValid
          ? "Code is valid"
          : "Error: Missing core Three.js components",
      });

      return isValid
        ? "Code is valid"
        : "Error: Missing core Three.js components";
    },
  }),
  new DynamicStructuredTool({
    name: "simple_executor",
    description: "Execute basic Three.js code",
    schema: z.object({
      code: z.string().describe("Simplified Three.js initialization code"),
    }),
    func: async ({ code }: { code: string }): Promise<string> => {
      console.log("Executing Three.js code:", typeof code === 'string' ? code.substring(0, 100) + "..." : 'Non-string content');

      // Send code to frontend for execution via WebSocket
      sendToClient(ws, {
        type: "execute",
        code,
        requestId: Date.now().toString(),
      });

      return "Code has been sent to frontend for execution via WebSocket, please check the 3D rendering area";
    },
  }),
  new DynamicStructuredTool({
    name: "generate_threejs_code",
    description: "Generate complete Three.js code based on description",
    schema: z.object({
      description: z
        .string()
        .describe("Natural language description of 3D scene"),
      complexity: z
        .enum(["simple", "medium", "complex"])
        .optional()
        .describe("Code complexity"),
    }),
    func: async ({
      description,
      complexity = "medium",
    }: {
      description: string;
      complexity?: "simple" | "medium" | "complex";
    }): Promise<string> => {
      // Can call previous code generation chain here
      const llm = createAzureLLM;
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `You are a professional Three.js code generation expert. Please generate complete HTML file code based on user description, including a complete Three.js scene.
Complexity: ${complexity}
Code requirements:

1. **使用现代ES模块方式导入Three.js**：
   - 使用importmap配置模块路径
   - 使用ES Module Shims确保兼容性
   - 避免使用已弃用的build/three.js和build/three.min.js
   - 将script标签改为type="module"
   - 使用ES模块方式导入Three.js和OrbitControls
   - 添加了ES Module Shims库，提高浏览器兼容性
   -"imports": {
    "three": "https://unpkg.com/three@0.153.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.153.0/examples/jsm/"
    - Three.js现在默认将OrbitControls作为模块导出 Stack Overflow，导出语句：import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

}

2. **完整的场景设置**：
   - 场景、相机、渲染器的基本配置


3. **交互控制**：
   - 使用OrbitControls实现旋转、缩放功能


5. **代码结构**：
   - 单HTML文件实现，便于在线平台直接运行
   - 内联CSS样式
   - 清晰的代码注释
6.例子：<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Apple Model with Three.js</title>
    <style>
        body { margin: 0; }
        canvas { display: block; }
    </style>
    <!-- ES Module Shims for compatibility -->
    <script async src="https://unpkg.com/es-module-shims@1.5.0/dist/es-module-shims.js"></script>
    
    <!-- Importmap for Three.js modules -->
    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.153.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.153.0/examples/jsm/"
        }
    }
    </script>
</head>
<body>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);

        // Camera setup
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 1, 3);

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // OrbitControls setup
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true; // Enable damping for smoother controls
        controls.dampingFactor = 0.25;

        // Apple geometry and material
        const appleGeometry = new THREE.SphereGeometry(0.5, 32, 32);
        const appleMaterial = new THREE.MeshStandardMaterial({ color: 0xff0000 });
        const apple = new THREE.Mesh(appleGeometry, appleMaterial);
        scene.add(apple);

        // Stem geometry and material
        const stemGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.3, 32);
        const stemMaterial = new THREE.MeshStandardMaterial({ color: 0x8b4513 });
        const stem = new THREE.Mesh(stemGeometry, stemMaterial);
        stem.position.y = 0.65;
        scene.add(stem);

        // Light setup
        const ambientLight = new THREE.AmbientLight(0x404040, 2); // Soft white light
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(5, 5, 5);
        scene.add(pointLight);

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update(); // Update controls
            
            // Add simple rotation animation
            apple.rotation.y += 0.01;
            
            renderer.render(scene, camera);
        }

        animate();

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
`,
        ],
        ["human", "{input}"],
      ]);

      const chain = prompt.pipe(llm);

      // Notify frontend that code generation has started
      sendToClient(ws, {
        type: "generation_started",
        description,
      });

      const result = await chain.invoke({ input: description });
      const generatedCode = typeof result.content === 'string' ? result.content : JSON.stringify(result.content);

      console.log("Generated code:", typeof generatedCode === 'string' ? generatedCode.substring(0, 200) + "..." : 'Non-string content');

      // Send generated code to frontend
      sendToClient(ws, {
        type: "code_generated",
        code: generatedCode,
        requestId: Date.now().toString(),
      });

      // Immediately send execute command to ensure code is executed
      sendToClient(ws, {
        type: "execute",
        code: generatedCode,
        requestId: Date.now().toString() + 1,
      });

      return typeof generatedCode === 'string' ? generatedCode : JSON.stringify(generatedCode);
    },
  }),
];

// Agent building layer
const createGenerationAgent = (llm: AzureChatOpenAI) => {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a Three.js code generation expert, generating basic code including Scene/Camera/Renderer
      1. **使用现代ES模块方式导入Three.js**：
   - 使用importmap配置模块路径
   - 使用ES Module Shims确保兼容性
   - 避免使用已弃用的build/three.js和build/three.min.js
   -使用0.158.0版本threejs
   - 将script标签改为type="module"
   - 使用ES模块方式导入Three.js和OrbitControls
   - 添加了ES Module Shims库，提高浏览器兼容性
   - 完整的场景设置：场景、相机、渲染器的基本配置`
    ],
    ["human", "{input}"],
  ]);
  return prompt.pipe(llm);
};

const createValidationAgent = (
  llm: AzureChatOpenAI,
  tools: DynamicStructuredTool[]
) => {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", `You are responsible for code validation and correction
    1. **使用现代ES模块方式导入Three.js**：
   - 使用importmap配置模块路径
   - 使用ES Module Shims确保兼容性
   - 避免使用已弃用的build/three.js和build/three.min.js
   -使用0.153.0版本threejs
   - 将script标签改为type="module"
   - 使用ES模块方式导入Three.js和OrbitControls
   - 添加了ES Module Shims库，提高浏览器兼容性`],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);
  return createToolCallingAgent({ llm, tools, prompt });
};

// Conversation Agent definition
const createConversationAgent = (
  llm: AzureChatOpenAI,
  tools: DynamicStructuredTool[]
) => {
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a professional Three.js assistant who can understand user needs and generate corresponding 3D scene code. If a user asks how to create specific 3D objects or scenes,
please use the generate_threejs_code tool to generate complete code. You can also use the code_validator tool to check code validity, or use the
simple_executor tool to execute code. If the user is just chatting, you can answer directly without using tools.
Please ensure the generated code can correctly render 3D scenes in browsers.`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  return createToolCallingAgent({ llm, tools, prompt });
};

// Workflow integration - modified to use WebSocket
export const createSimpleWorkflow = (ws: ExtendedWebSocket) => {
  const llm = createAzureLLM;
  const tools = createBasicTools(ws);

  // Build chain process
  const generationChain = createGenerationAgent(llm);
  const validationAgent = createValidationAgent(llm, tools);

  const executor = new AgentExecutor({
    agent: validationAgent,
    tools,
    verbose: true,
    tags: ["threejs-agent", "code-generation"], // Add tags for LangSmith filtering
  });

  // Combined sequential workflow
  /**
 * Asynchronous function that orchestrates a two-phase workflow for code generation and validation.
 * 
 * @param input - The input string used to drive the generation process.
 * 
 * The function notifies the frontend at various stages of the workflow:
 * 1. It starts by notifying the frontend of the workflow's initiation for code generation.
 * 2. In Phase 1, it generates code by invoking the `generationChain` with the input and configuration.
 * 3. It then notifies the frontend upon completion of code generation.
 * 4. The function transitions to Phase 2 by notifying the frontend of the validation phase start.
 * 5. It invokes the `executor` to validate and execute the generated code.
 * 6. Finally, it notifies the frontend of the workflow's completion and returns an object containing the raw generated code and the validation result.
 */
  return async (input: string) => {
    // Notify frontend of workflow start
    sendToClient(ws, {
      type: "workflow_started",
      stage: "generation",
    });

    // Phase 1: Generate code
    const generated = await generationChain.invoke({
      input,
      configurable: {
        tags: ["code-generation"],
        metadata: {
          stage: "code-generation",
          inputType: "prompt",
          projectName: LANGCHAIN_PROJECT,
        },
      },
    });
    const rawCode = typeof generated.content === 'string' ? generated.content : JSON.stringify(generated.content);

    // Notify frontend of code generation completion
    sendToClient(ws, {
      type: "generation_completed",
      code: rawCode,
    });

    // Notify frontend of validation phase start
    sendToClient(ws, {
      type: "workflow_started",
      stage: "validation",
    });

    // Phase 2: Validate and execute
    const result = await executor.invoke({
      input: `Validate and execute the following code:\n${rawCode}`,
      chat_history: [
        /* Can pass history messages here */
      ],
      configurable: {
        tags: ["code-validation"],
        metadata: {
          stage: "code-validation",
          projectName: LANGCHAIN_PROJECT,
        },
      },
    });

    // Notify frontend of workflow completion
    sendToClient(ws, {
      type: "workflow_completed",
      result: result.output,
    });

    return {
      generatedCode: rawCode,
      validationResult: result.output,
    };
  };
};

// Conversational Agent workflow - modified to use WebSocket
export const createConversationalWorkflow = (ws: ExtendedWebSocket) => {
  const llm = createAzureLLM;
  const tools = createBasicTools(ws);
  const memory = new BufferMemory({
    returnMessages: true,
    memoryKey: "chat_history",
    inputKey: "input",
    outputKey: "output",
  });

  // Create conversational Agent
  const conversationAgent = createConversationAgent(llm, tools);

  // Create executor
  const executor = new AgentExecutor({
    agent: conversationAgent,
    tools,
    memory,
    verbose: true,
    tags: ["threejs-agent", "conversation"], // Add tags for LangSmith filtering
  });

  // Return traced conversation function
  return async (input: string) => {
    // Notify frontend that AI is processing request
    sendToClient(ws, {
      type: "ai_thinking",
      message: "AI is processing your request...",
    });

    const result = await executor.invoke({
      input,
      configurable: {
        tags: ["conversation-agent"],
        metadata: {
          conversationId: Date.now().toString(),
          environment: process.env.NODE_ENV || "development",
          inputType: "natural-language",
          projectName: LANGCHAIN_PROJECT,
        },
      },
    });

    // Check if code was executed
    let generatedCode = null;

    // Add conversation history to memory
    await memory.saveContext({ input }, { output: result.output });

    return {
      response: result.output,
      generatedCode,
    };
  };
};

// CLI interface - add WebSocket server start
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function conversationLoop(): Promise<void> {
  console.log("Three.js Conversation Assistant started");
  console.log(`WebSocket server running, port: ${PORT}`);
  console.log("Waiting for client connections...");

  // Start WebSocket server
  server.listen(PORT, () => {
    console.log(`WebSocket server started, listening on port ${PORT}`);
  });

  // Add CLI commands for server control
  rl.question(
    "\nEnter 'exit' to quit server, 'clients' to view connection count, 'restart' to restart server: ",
    async (input) => {
      if (input.toLowerCase() === "exit") {
        console.log("Shutting down server...");
        clearInterval(heartbeatInterval);
        server.close();
        rl.close();
        process.exit(0);
      } else if (input.toLowerCase() === "clients") {
        console.log(`Current connected clients: ${activeConnections.size}`);
        conversationLoop();
      } else if (input.toLowerCase() === "restart") {
        console.log("Restarting server...");
        server.close(() => {
          server.listen(PORT, () => {
            console.log(`Server has restarted, listening on port ${PORT}`);
            conversationLoop();
          });
        });
      } else {
        console.log("Unknown command");
        conversationLoop();
      }
    }
  );
}

// Handle process termination signals
process.on("SIGINT", () => {
  console.log("\nShutting down server...");
  clearInterval(heartbeatInterval);
  server.close();
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("\nShutting down server...");
  clearInterval(heartbeatInterval);
  server.close();
  process.exit(0);
});

// Start server
conversationLoop();
