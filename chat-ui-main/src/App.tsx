import React, { useState, useEffect } from 'react';
import { MessageSquarePlus, Menu, Settings, LogOut, Send, Bot, User } from 'lucide-react';
import hljs from "highlight.js";
import "highlight.js/styles/github-dark.css"; // You can choose another theme
import "katex/dist/katex.min.css";
import katex from "katex";

interface Message {
  id: number;
  content: string;
  isBot: boolean;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, content: "Hello! How can I help you today?", isBot: true }
  ]);
  const [input, setInput] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [model, setModel] = useState("gpt-4o");
  const [isLoading, setIsLoading] = useState(false);

  // Connect to log stream
  useEffect(() => {
    const eventSource = new EventSource('http://localhost:5000/api/logs');
    eventSource.onmessage = (event) => {
      setLogs((prevLogs) => [...prevLogs, event.data]);
    };

    eventSource.onerror = (error) => {
      console.error('Log stream error:', error);
      eventSource.close();
    };

    return () => eventSource.close();
  }, []);

  // Highlight code whenever messages update
  useEffect(() => {
    hljs.highlightAll(); // Apply syntax highlighting
  }, [messages]); // Runs every time a new AI message is added


  const formatResponse = (text: string) => {
    // ✅ Check if the response is JSON format before parsing
    if (text.trim().startsWith("{") && text.trim().endsWith("}")) {
      try {
        const json = JSON.parse(text);
  
        // ✅ Start with the message if it exists
        let formattedMessage = json.message || "";
  
        // ✅ If response contains an image URL, append an <img> tag
        if (json.image_url) {
          formattedMessage += `<br><img src="http://localhost:5000${json.image_url}" alt="Visualization" class="plot-image" />`;
        }
  
        return formattedMessage;
      } catch (e) {
        console.error("❌ JSON Parse Error:", e);
      }
    }
  
    // ✅ Handle Markdown links and replace them with images
    const markdownImageRegex = /\[here\]\((.*?)\)/g;
  
    // Replace Markdown link with <img> tag
    text = text.replace(markdownImageRegex, (match, imageUrl) => {
      // Remove 'sandbox:' prefix if present
      imageUrl = imageUrl.replace('sandbox:', '');
      return `<br><img src="http://localhost:5000${imageUrl}" alt="Visualization" class="plot-image" />`;
    });

    // ✅ NEW: Handle Markdown image syntax ![alt](...)
    const markdownImageSyntaxRegex = /!\[.*?\]\((.*?)\)/g;
    text = text.replace(markdownImageSyntaxRegex, (match, imageUrl) => {
      return `<br><img src="http://localhost:5000${imageUrl}" alt="Visualization" class="plot-image" />`;
    });
    
    return text
      // ✅ Keep the rest of your existing formatting logic as it is
      .replace(/!\[.*?\]\((data:image\/png;base64,[^\)]+)\)/g, '<img src="$1" alt="Visualization" class="max-w-full rounded-lg shadow-md"/>')
      .replace(/```(\w+)?\n([\s\S]+?)```/g, (_, lang, code) => {
        const language = lang || "plaintext";
        return `
          <div class="code-container">
            <div class="code-header">
              <span>${language.toUpperCase()}</span>
            </div>
            <pre><code class="hljs ${language}">${code
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/\n/g, "<br>")
              .replace(/ /g, "&nbsp;")
            }</code></pre>
          </div>`;
      })
      .replace(/`([^`]+)`/g, '<code class="bg-gray-200 text-red-600 px-1 rounded">$1</code>')
      .replace(/### (.*?)\n/g, '<h2 class="text-lg font-bold mt-4">$1</h2>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n\d+\.\s(.*?)$/gm, '<li class="ml-4 list-decimal">$1</li>')
      .replace(/(<li class="ml-4 list-decimal">.*?<\/li>)+/g, '<ol class="ml-4 list-decimal">$&</ol>')
      .replace(/\n-\s(.*?)$/gm, '<li class="ml-4 list-disc">$1</li>')
      .replace(/(<li class="ml-4 list-disc">.*?<\/li>)+/g, '<ul class="ml-4 list-disc">$&</ul>')
      .replace(/\n(?!<\/?pre>)/g, "<br>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 underline hover:text-blue-800">$1</a>')
      .replace(/\\\[(.*?)\\\]/gs, (_, equation) => {
        return `<div class="math-block">${katex.renderToString(equation, { throwOnError: false, displayMode: true })}</div>`;
      })
      .replace(/\\\((.*?)\\\)/g, (_, equation) => {
        return `<span class="math-inline">${katex.renderToString(equation, { throwOnError: false, displayMode: false })}</span>`;
      });
  };
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { id: Date.now(), content: input, isBot: false };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input, model }),
      });

      const data = await response.json();
      const botResponse = data.response ? data.response : "⚠️ Error: Invalid response format";
      const formattedResponse = formatResponse(botResponse);
      const botMessage = { id: Date.now() + 1, content: formattedResponse, isBot: true };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }

    setInput("");
  };

  const handleEndChat = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/save_memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
  
      const data = await response.json();
      setLogs((prevLogs) => [...prevLogs, `🛠️ MEMORY: ${data.message}`]);
  
      if (data.status === "success") {
        setMessages([]); // Clear messages
        setLogs((prevLogs) => [...prevLogs, "✅ Chat ended. Memory saved."]);
      }
    } catch (error) {
      setLogs((prevLogs) => [...prevLogs, `❌ MEMORY: Error saving memory.`]);
    }
  };
  
  // Determine log style based on source
  const getLogStyle = (log: string) => {
    if (log.includes("[SYSTEM]")) return "text-gray-500";
    if (log.includes("[AGENT]")) return "text-blue-600";
    if (log.includes("[LLM]")) return "text-green-600";
    if (log.includes("[MEMORY]")) return "text-magenta-600";
    return "text-black";
  };

  const getLogIcon = (log: string) => {
    if (log.includes("[SYSTEM]")) return "⚙️";
    if (log.includes("[AGENT]")) return "🤖";
    if (log.includes("[LLM]")) return "🧠";
    if (log.includes("[MEMORY]")) return "🛠️";
    return "🔹";
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <div className={`${isSidebarOpen ? 'w-64' : 'w-0'} bg-gray-900 transition-all duration-300 overflow-hidden`}>
        <div className="p-4">
          <button className="w-full text-white bg-gray-700 hover:bg-gray-600 rounded-md p-3 flex items-center gap-3">
            <MessageSquarePlus size={16} />
            New Chat
          </button>
        </div>
        <div className="absolute bottom-0 left-0 w-64 p-4 border-t border-gray-700">
          <button className="w-full text-gray-300 hover:bg-gray-700 rounded-md p-2 flex items-center gap-3">
            <Settings size={16} />
            Settings
          </button>
          <button className="w-full text-gray-300 hover:bg-gray-700 rounded-md p-2 flex items-center gap-3">
            <LogOut size={16} />
            Log out
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <header className="bg-white border-b border-gray-200 p-4 flex items-center justify-between">
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-gray-100 rounded-md">
            <Menu size={20} />
          </button>

          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="border p-2 rounded-lg"
          >
            <option value="gpt-4o">      (Online LLM GPT4-o)</option>
            <option value="gpt-4-turbo"> (Online LLM GPT4-turbo)</option>
            <option value="llama3.2">    (Local LLM Llama3.2)</option>
            <option value="deepseek-r1"> (Local LLM DeepSeek-R1)</option>
          </select>
        </header>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map(message => (
            <div key={message.id} className={`flex items-start gap-4 ${message.isBot ? 'bg-gray-50' : 'bg-white'} p-6`}>
              <div className={`p-2 rounded-md ${message.isBot ? 'bg-green-500' : 'bg-gray-900'}`}>
                {message.isBot ? <Bot className="text-white" size={20} /> : <User className="text-white" size={20} />}
              </div>
              <div className="flex-1">
                {message.isBot ? (
                  <div dangerouslySetInnerHTML={{ __html: message.content }} className="text-gray-800 leading-relaxed space-y-4"/>
                ) : (
                  <p className="text-gray-800 leading-relaxed">{message.content}</p>
                )}
              </div>
            </div>
          ))}
        </div>
        {/*Typing Indicator */}
        {isLoading && (
        <div className="flex items-center space-x-2 p-4">
          <div className="w-3 h-3 bg-gray-500 rounded-full animate-bounce"></div>
          <div className="w-3 h-3 bg-gray-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
          <div className="w-3 h-3 bg-gray-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
          <span className="text-gray-500">Typing...</span>
        </div>
        )}
        {/* Logs Section */}
        <div className="log-window">
          <h3 className="text-lg font-bold mb-2">🛠 Real-Time Logs</h3>
          <ul className="log-messages">
            {logs.map((log, index) => (
              <li key={index} className={`log-message ${getLogStyle(log)}`}>
                {getLogIcon(log)} {log}
              </li>
            ))}
          </ul>
        </div>
        <div className="border-t border-gray-200 p-4 bg-white">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Send a message..."
              className="w-full p-4 pr-12 rounded-lg border border-gray-200 focus:border-gray-300 focus:ring focus:ring-gray-200 focus:ring-opacity-50"
            />
            <button type="submit" className="absolute right-4 top-1/2 -translate-y-1/2 p-2 text-gray-400 hover:text-gray-600">
              <Send size={20} />
            </button>
          </form>
          {/* End Chat Button */}
                    <button 
            onClick={handleEndChat} 
            className="ml-4 px-6 py-3 bg-blue-400 text-white rounded-lg hover:bg-blue-500"
          >
            End Chat
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
