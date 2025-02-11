import React, { useState } from 'react';
import { MessageSquarePlus, Menu, Settings, LogOut, Send, Bot, User } from 'lucide-react';

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
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [model, setModel] = useState("llama3.2"); // Default model is Ollama

  const formatResponse = (text: string) => {
    return text
      .replace(/### (.*?)\n/g, '<h2 class="text-lg font-bold mt-4">$1</h2>') // Convert headers
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Convert **bold** to <strong>
      
      // ✅ Fix numbered lists (1. Item)
      .replace(/\n\d+\.\s(.*?)$/gm, '<li class="ml-4 list-decimal">$1</li>') 
      .replace(/(<li class="ml-4 list-decimal">.*?<\/li>)+/g, '<ol class="ml-4 list-decimal">$&</ol>') 
      
      // ✅ Fix bullet point lists (- Item)
      .replace(/\n-\s(.*?)$/gm, '<li class="ml-4 list-disc">$1</li>') 
      .replace(/(<li class="ml-4 list-disc">.*?<\/li>)+/g, '<ul class="ml-4 list-disc">$&</ul>') 
      
      .replace(/\n/g, "<br>"); // Convert remaining new lines to <br> for better readability
  };


  // Function to send message to the backend
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
  
    const userMessage = { id: Date.now(), content: input, isBot: false };
    setMessages((prev) => [...prev, userMessage]);
  
    try {
      const response = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input, model }),
      });
  
      const data = await response.json();
      const botResponse = data.response ? data.response : "⚠️ Error: Invalid response format";
  
      // ✅ Apply improved formatting before rendering
      const formattedResponse = formatResponse(botResponse);
  
      const botMessage = { id: Date.now() + 1, content: formattedResponse, isBot: true };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
    }
  
    setInput("");
  };
  
  

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
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

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 p-4 flex items-center justify-between">
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-gray-100 rounded-md">
            <Menu size={20} />
          </button>

          {/* Dropdown for selecting AI model */}
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="border p-2 rounded-lg"
          >
            <option value="llama3.2">    (Local LLM Llama3.2)</option>
            <option value="deepseek-r1"> (Local LLM DeepSeek-R1)</option>
            <option value="gpt-4o">      (Online LLM GPT4-o)</option>
            <option value="gpt-4-turbo"> (Online LLM GPT4-turbo)</option>
          </select>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map(message => (
            <div key={message.id} className={`flex items-start gap-4 ${message.isBot ? 'bg-gray-50' : 'bg-white'} p-6`}>
              <div className={`p-2 rounded-md ${message.isBot ? 'bg-green-500' : 'bg-gray-900'}`}>
                {message.isBot ? <Bot className="text-white" size={20} /> : <User className="text-white" size={20} />}
              </div>
              <div className="flex-1">
              {message.isBot ? (
                <div dangerouslySetInnerHTML={{ __html: message.content }} className="text-gray-800 leading-relaxed"></div>
              ) : (
                <p className="text-gray-800 leading-relaxed">{message.content}</p>
              )}
            </div>
            </div>
          ))}
        </div>

        {/* Input Field */}
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
        </div>
      </div>
    </div>
  );
}

export default App;
