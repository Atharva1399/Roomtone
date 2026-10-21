// The pipeline's rule checks, said in plain language with the number that decided them.
// Raw details look like "wall vs floor: dE 13.9 (need >= 8 so they don't blend)".

interface RawCheck {
  rule: string
  passed: boolean
  detail: string
}

const nums = (s: string) => (s.match(/\d+(?:\.\d+)?/g) ?? []).map(Number)
const r1 = (n: number | undefined) => (n === undefined ? '' : Number.isInteger(n) ? String(n) : n.toFixed(1))

export function checkSentence(c: RawCheck): string {
  const n = nums(c.detail)
  switch (c.rule) {
    case 'wall-distinct-from-anchor': {
      const thing = c.detail.match(/wall vs (\w+)/)?.[1] ?? 'furniture'
      return c.passed
        ? `Walls stand apart from your ${thing}: colour difference ${r1(n[0])}, where 8 is the minimum.`
        : `Walls are close to your ${thing}: colour difference ${r1(n[0])}, under the minimum of 8.`
    }
    case 'accent-stands-out':
      return c.passed
        ? `The accent reads clearly against the walls: difference ${r1(n[0])}, needs ≥ 20.`
        : `The accent is too close to the walls: difference ${r1(n[0])}, needs ≥ 20.`
    case 'wall-lightness-fits-goal':
      return c.passed
        ? `Wall lightness ${r1(n[0])} sits in this goal's range of ${r1(n[1])} to ${r1(n[2])}.`
        : `Wall lightness ${r1(n[0])} is outside this goal's range of ${r1(n[1])} to ${r1(n[2])}.`
    case 'wall-not-garish':
      return c.passed
        ? `Colour strength ${r1(n[0])} is calm enough for a whole wall (limit ${r1(n[1])}).`
        : `Colour strength ${r1(n[0])} is strong for a whole wall (limit ${r1(n[1])}).`
    case 'dim-room-light-wall':
      return c.passed
        ? `Light enough for a dim room: it reflects ${r1(n[0])}% of light, needs ≥ 50%.`
        : `Dark for a dim room: it reflects ${r1(n[0])}% of light, under 50%.`
    case 'harmony-with-furnishings':
      return c.passed
        ? `Hues sit in harmony with what is already in the room (clash score ${r1(n[0])} of 1).`
        : `Some furnishings clash in hue with these walls (clash score ${r1(n[0])} of 1).`
    case 'real-paint-close':
      return c.passed
        ? `Real paints land within ${r1(n[0])} of the target colour; under 1 is invisible to the eye.`
        : `The nearest real paints are ${r1(n[0])} away from the target colour.`
    default:
      return c.detail.replace(/>=/g, '≥')
  }
}
