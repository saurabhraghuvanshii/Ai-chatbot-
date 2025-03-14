import React, { useState } from 'react';
import { User, Send, MessageSquare } from 'lucide-react';
import jenkins from "./assets/jenkins.svg"

interface Message {
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      text: "Hello! I'm Jenkins AI Assistant. How can I help you with your CI/CD pipeline today?",
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessage: Message = {
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages([...messages, newMessage]);
    setInput('');

    // Simulate bot response
    setTimeout(() => {
      const botResponse: Message = {
        text: "I'm processing your request. As an AI assistant, I'm here to help with Jenkins-related queries.",
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botResponse]);
    }, 1000);
  };

  return (
    <div className="flex h-screen bg-[#1c1c1c] text-white">
      {/* Left Sidebar */}
      <div className="w-[260px] bg-[#2c2c2c] border-r border-[#333] flex flex-col">
        {/* Jenkins Logo and Title */}
        <div className="p-4 border-b border-[#333]">
          <div className="flex items-center space-x-3">
            <img 
              src={jenkins} 
              alt="Jenkins Logo" 
              className="w-8 h-8"
            />
            <span className="text-lg font-bold">Jenkins AI Assistant</span>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <button className="w-full bg-[#D33833] text-white rounded-lg py-2 px-4 flex items-center justify-center space-x-2 hover:bg-[#B32D29] transition-colors">
            <MessageSquare className="w-4 h-4" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-2">
          <div className="text-sm text-gray-400 px-2 py-1">Recent chats</div>
          <button className="w-full text-left px-3 py-2 text-gray-300 hover:bg-[#333] rounded-lg transition-colors">
            CI/CD Pipeline Setup
          </button>
          <button className="w-full text-left px-3 py-2 text-gray-300 hover:bg-[#333] rounded-lg transition-colors">
            Jenkins Configuration
          </button>
        </div>

        {/* Build Info */}
        <div className="p-4 border-t border-[#333] text-xs text-gray-400">
          <p>Build great things at any scale</p>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex items-start space-x-2 ${
                message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}
            >
              <div className={`flex-shrink-0 ${
                message.sender === 'user' ? 'bg-[#D33833] p-2 rounded-full' : ''
              }`}>
                {message.sender === 'user' ? (
                  <User className="w-6 h-6 text-white" />
                ) : (
                  <div className="w-10 h-10 rounded-full bg-[#335061] p-1">
                    <img 
                      src={jenkins}
                      alt="Jenkins Butler"
                      className="w-full h-full"
                    />
                  </div>
                )}
              </div>
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  message.sender === 'user'
                    ? 'bg-[#D33833]'
                    : 'bg-[#335061]'
                }`}
              >
                <p className="text-white">{message.text}</p>
                <span className="text-xs text-gray-300 mt-1 block">
                  {message.timestamp.toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Input Form */}
        <div className="border-t border-[#333] p-4">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex space-x-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message here..."
                className="flex-1 bg-[#1c1c1c] text-white p-3 rounded-lg border border-[#333] focus:outline-none focus:border-[#D33833] transition-colors"
              />
              <button
                type="submit"
                className="bg-[#D33833] text-white px-6 py-3 rounded-lg hover:bg-[#B32D29] transition-colors flex items-center space-x-2 font-medium"
              >
                <span>Send</span>
                <Send className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}


export default App;