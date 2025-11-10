'use client';

import Chat from '.';
import { ExampleChatService } from '@chatscope/use-chat/dist/examples';
import { AutoDraft } from '@chatscope/use-chat/dist/enums/AutoDraft';
import {
  BasicStorage,
  ChatMessage,
  ChatProvider,
  IStorage,
  MessageContentType,
  UpdateState,
} from '@chatscope/use-chat';
import { nanoid } from 'nanoid';

export default function RootLayout({}: Readonly<{
  children: React.ReactNode;
}>) {
  const messageIdGenerator = (message: ChatMessage<MessageContentType>) =>
    nanoid();
  const groupIdGenerator = () => nanoid();

  const serviceFactory = (storage: IStorage, updateState: UpdateState) => {
    return new ExampleChatService(storage, updateState);
  };

  const akaneStorage = new BasicStorage({
    groupIdGenerator,
    messageIdGenerator,
  });

  return (
    <ChatProvider
      serviceFactory={serviceFactory}
      storage={akaneStorage}
      config={{
        typingThrottleTime: 250,
        typingDebounceTime: 900,
        debounceTyping: true,
        autoDraft: AutoDraft.Save | AutoDraft.Restore,
      }}
    >
      <Chat />
    </ChatProvider>
  );
}
