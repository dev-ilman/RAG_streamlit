import { afterEach, describe, expect, it, vi } from 'vitest';

import { askStream } from '@/api/chat';
import { setAccessToken } from '@/api/client';

function sseResponse(frames: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      // Split mid-frame to prove the parser reassembles across chunks.
      const payload = frames.join('');
      const half = Math.floor(payload.length / 2);
      controller.enqueue(encoder.encode(payload.slice(0, half)));
      controller.enqueue(encoder.encode(payload.slice(half)));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  });
}

const frame = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

describe('askStream', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('dispatches meta, token and done events in order', async () => {
    setAccessToken('token-abc');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        sseResponse([
          frame('meta', {
            session_id: 's1',
            citations: [{ chunk_id: 'c1', document_title: 'Doc' }],
          }),
          frame('token', { text: 'Lithium ' }),
          frame('token', { text: 'plating.' }),
          frame('done', { message_id: 'm1', latency_ms: 42 }),
        ]),
      ),
    );

    const tokens: string[] = [];
    let citationCount = 0;
    let latency = 0;
    const onError = vi.fn();

    await askStream('s1', 'why?', {
      onMeta: (citations) => (citationCount = citations.length),
      onToken: (text) => tokens.push(text),
      onDone: (ms) => (latency = ms),
      onError,
    });

    expect(citationCount).toBe(1);
    expect(tokens.join('')).toBe('Lithium plating.');
    expect(latency).toBe(42);
    expect(onError).not.toHaveBeenCalled();
  });

  it('surfaces a server-sent error frame', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(sseResponse([frame('error', { message: 'upstream down' })])),
    );

    const onError = vi.fn();
    await askStream('s1', 'why?', { onError });

    expect(onError).toHaveBeenCalledWith('upstream down');
  });

  it('reports a non-200 response instead of hanging', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 500 })));

    const onError = vi.fn();
    await askStream('s1', 'why?', { onError });

    expect(onError).toHaveBeenCalledWith('Request failed with status 500');
  });

  it('ignores malformed JSON frames rather than throwing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        sseResponse(['event: token\ndata: {not json}\n\n', frame('token', { text: 'ok' })]),
      ),
    );

    const tokens: string[] = [];
    await askStream('s1', 'why?', { onToken: (t) => tokens.push(t) });

    expect(tokens).toEqual(['ok']);
  });
});
