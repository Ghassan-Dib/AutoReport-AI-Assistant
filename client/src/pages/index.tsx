'use client';
import { useState } from 'react';
import Chat from '../components/Chat';
import {
  ChatMessage,
  MessageContent,
  MessageContentType,
  MessageDirection,
  MessageStatus,
  TextContent,
} from '@chatscope/use-chat';

const App = () => {
  const greetingMessage = new ChatMessage({
    id: '',
    senderId: '',
    status: MessageStatus.Sent,
    direction: MessageDirection.Incoming,
    contentType: MessageContentType.TextMarkdown,
    content:
      'Hello, I’m your <Strong>Automotive Annual Report</Strong> Analyst Assistant. How can I help you?' as unknown as MessageContent<TextContent>,
  });

  const [selectedType, setSelectedType] = useState<
    'standard' | 'multi_query' | 'query_decomposition'
  >('standard');

  const [messages, setMessages] = useState<ChatMessage<MessageContentType>[]>([
    greetingMessage,
  ]);

  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (message: String) => {
    const newMessage = new ChatMessage({
      id: '',
      senderId: '',
      content: message as unknown as MessageContent<TextContent>,
      status: MessageStatus.Sent,
      direction: MessageDirection.Outgoing,
      contentType: MessageContentType.TextMarkdown,
    });

    const newMessages = [...messages, newMessage];

    setMessages(newMessages);
    setIsTyping(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: message,
          retriever_type: selectedType,
        }),
      });
      const data = await response.json();
      data.direction = MessageDirection.Incoming;
      data.contentType = MessageContentType.TextMarkdown;

      // add sources to the message content if available
      if (data.sources && Array.isArray(data.sources)) {
        let sourcesContent = '<br/><br/><Strong>Sources:</Strong><ul>';
        data.sources.forEach((source: string) => {
          sourcesContent += `<li>${source}</li>`;
        });
        sourcesContent += '</ul>';
        data.content += sourcesContent;
      }

      setMessages((prevMessages: any) => [...prevMessages, data]);
      setIsTyping(false);
    } catch (error) {
      console.error(error);
      setIsTyping(false);
    }
  };

  return (
    <Chat
      messages={messages}
      isTyping={isTyping}
      onSelectType={setSelectedType}
      onSend={handleSend}
    />
  );
};

export default App;
