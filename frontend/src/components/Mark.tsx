// The Roomtone mark: a wall split 60 / 30 / 10, the rule every palette is built on.
export function Mark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <rect x="0" y="0" width="14.4" height="24" fill="var(--plaster)" />
      <rect x="14.4" y="0" width="7.2" height="24" fill="var(--walnut)" />
      <rect x="21.6" y="0" width="2.4" height="24" fill="var(--rust)" />
    </svg>
  )
}

export function Wordmark() {
  return (
    <span className="wordmark">
      <Mark />
      <span>Roomtone</span>
    </span>
  )
}
