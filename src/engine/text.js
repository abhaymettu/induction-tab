// The tab's alphabet: printable ASCII, 96 tokens. Token t is the character with code t + 32; 95 (DEL) never renders.
export const encode = (text) => Uint32Array.from([...text].map((ch) => {
  const t = ch.charCodeAt(0) - 32;
  return t >= 0 && t < 95 ? t : 0;
}));
export const decode = (t) => (t >= 0 && t < 95 ? String.fromCharCode(t + 32) : ' ');
