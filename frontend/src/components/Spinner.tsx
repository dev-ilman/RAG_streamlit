export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <>
      <span className="spinner" role="status" aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </>
  );
}
