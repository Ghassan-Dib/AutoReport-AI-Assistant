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

interface ChatProps {
  messages: ChatMessage<MessageContentType>[];
  isTyping: boolean;
  onSend: (message: String) => void;
}

const Chat: React.FC<ChatProps> = ({ messages, isTyping, onSend }) => {
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
