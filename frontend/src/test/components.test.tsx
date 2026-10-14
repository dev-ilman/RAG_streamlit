import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { CitationList } from '@/components/CitationList';
import { MessageBubble } from '@/components/MessageBubble';
import { StatusBadge } from '@/components/StatusBadge';
import type { ChatMessage, Citation } from '@/types';

const citation: Citation = {
  chunk_id: 'c1',
  document_id: 'd1',
  document_title: 'Battery ageing',
  chunk_index: 3,
  score: 0.8421,
  excerpt: 'Solid electrolyte interphase growth consumes cyclable lithium.',
};

const assistantMessage: ChatMessage = {
  id: 'm1',
  session_id: 's1',
  role: 'assistant',
  content: 'SEI growth consumes lithium.',
  citations: [citation],
  model: 'gemini-2.5-flash',
  latency_ms: 730,
  created_at: new Date().toISOString(),
};

describe('StatusBadge', () => {
  it.each([
    ['ready', 'Ready'],
    ['failed', 'Failed'],
    ['pending', 'Queued'],
    ['processing', 'Indexing'],
  ] as const)('renders %s as %s', (status, label) => {
    render(<StatusBadge status={status} />);
    expect(screen.getByTestId('status-badge')).toHaveTextContent(label);
  });
});

describe('CitationList', () => {
  it('renders nothing when there are no citations', () => {
    const { container } = render(<CitationList citations={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('summarises the source count and shows the excerpt', () => {
    render(<CitationList citations={[citation]} />);
    expect(screen.getByText('1 source')).toBeInTheDocument();
    expect(screen.getByText('Battery ageing')).toBeInTheDocument();
    expect(screen.getByText(/0\.842/)).toBeInTheDocument();
  });
});

describe('MessageBubble', () => {
  it('shows the assistant answer with its sources and latency', () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText('Assistant')).toBeInTheDocument();
    expect(screen.getByText('SEI growth consumes lithium.')).toBeInTheDocument();
    expect(screen.getByText(/730 ms/)).toBeInTheDocument();
  });

  it('omits citations for a user turn', () => {
    render(
      <MessageBubble
        message={{ ...assistantMessage, role: 'user', content: 'Why?', citations: [citation] }}
      />,
    );
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.queryByText('1 source')).not.toBeInTheDocument();
  });
});
