import React from 'react';
import {
  MainContainer,
  ChatContainer,
  Message,
  MessageList,
  MessageInput,
  TypingIndicator,
  ConversationHeader,
  Avatar,
  MessageSeparator,
} from '@chatscope/chat-ui-kit-react';
import { ChatMessage, MessageContentType } from '@chatscope/use-chat';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';

import avatar from '../assets/avatar.svg';

type ChatType = 'standard' | 'multi_query' | 'query_decomposition';

interface ChatTypeOption {
  label: string;
  value: ChatType;
}
interface ChatProps {
  messages: ChatMessage<MessageContentType>[];
  isTyping: boolean;
  onSelectType: (type: ChatType) => void;
  onSend: (message: String) => void;
}

const chatTypes: ChatTypeOption[] = [
  { label: 'Standard', value: 'standard' },
  { label: 'Multi-query', value: 'multi_query' },
  { label: 'Query Decomposition', value: 'query_decomposition' },
];

const Chat: React.FC<ChatProps> = ({
  messages,
  isTyping,
  onSelectType,
  onSend,
}) => {
  const [selectedType, setSelectedType] = React.useState<ChatType>('standard');

  const handleTypeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedType(event.target.value as ChatType);
    onSelectType(event.target.value as ChatType);
  };

  return (
    <MainContainer
      responsive
      style={{
        height: '1050px',
        width: '100%',
        borderRadius: '10px',
        border: '5px solid black',
        padding: '10px',
      }}
    >
      <ChatContainer>
        <ConversationHeader>
          <Avatar src={avatar.src} size='md' status='available' />
          <ConversationHeader.Content
            info='Active'
            userName='AutoReport Assistant'
          />
          <ConversationHeader.Actions>
            <select
              value={selectedType}
              onChange={handleTypeChange}
              style={{
                padding: '6px 12px',
                borderRadius: '4px',
                border: '1px solid #ccc',
                backgroundColor: 'white',
                cursor: 'pointer',
                fontSize: '14px',
              }}
            >
              {chatTypes.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
          </ConversationHeader.Actions>
        </ConversationHeader>
        <MessageList
          typingIndicator={
            isTyping && (
              <TypingIndicator
                content='searching documents..'
                style={{ fontSize: '1.5rem', padding: '10px 100px' }}
              />
            )
          }
          style={{ padding: '10px 100px' }}
        >
          <MessageSeparator
            content={new Date().toLocaleDateString('en-US', {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric',
            })}
          />
          {messages.map((message, i) => {
            return (
              <Message
                key={i}
                model={{
                  payload: message.content,
                  direction: message.direction,
                  position: 'single',
                }}
                style={{
                  fontSize: '1.5rem',
                }}
              />
            );
          })}
        </MessageList>
        <MessageInput
          attachButton={false}
          placeholder='ask your question here...'
          onSend={onSend}
          {...(isTyping && { disabled: true })}
          style={{
            fontSize: '1.5rem',
            padding: '10px 100px',
          }}
        />
      </ChatContainer>
    </MainContainer>
  );
};

export default Chat;
