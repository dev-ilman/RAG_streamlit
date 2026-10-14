import { CitationList } from '@/components/CitationList';
import type { ChatMessage } from '@/types';

interface Props {
  message: ChatMessage;
  /** Renders the blinking caret while tokens are still arriving. */
  streaming?: boolean;
}

export function MessageBubble({ message, streaming = false }: Props) {
  const isUser = message.role === 'user';
  return (
    <article className={`msg ${isUser ? 'user' : 'assistant'}`}>
      <div className="role">{isUser ? 'You' : 'Assistant'}</div>
      <div className={`body ${streaming ? 'caret' : ''}`}>{message.content}</div>
      {!isUser && <CitationList citations={message.citations} />}
      {!isUser && message.latency_ms !== null && message.latency_ms > 0 && (
        <div className="stamp">
          {message.model ?? 'model'} &middot; {message.latency_ms} ms
        </div>
      )}
    </article>
  );
}
