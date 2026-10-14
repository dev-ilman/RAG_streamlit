import type { Citation } from '@/types';

export function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;

  return (
    <details className="citations">
      <summary>
        {citations.length} source{citations.length === 1 ? '' : 's'}
      </summary>
      <ol>
        {citations.map((citation) => (
          <li key={citation.chunk_id}>
            <span className="src">{citation.document_title}</span>{' '}
            <span className="score">
              chunk {citation.chunk_index} &middot; {citation.score.toFixed(3)}
            </span>
            <div className="excerpt">{citation.excerpt}</div>
          </li>
        ))}
      </ol>
    </details>
  );
}
