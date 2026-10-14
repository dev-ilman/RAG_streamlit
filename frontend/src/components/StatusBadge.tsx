import type { DocumentStatus } from '@/types';

const LABELS: Record<DocumentStatus, string> = {
  pending: 'Queued',
  processing: 'Indexing',
  ready: 'Ready',
  failed: 'Failed',
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  return (
    <span className={`badge badge-${status}`} data-testid="status-badge">
      {LABELS[status]}
    </span>
  );
}
