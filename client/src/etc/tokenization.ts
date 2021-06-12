export interface Tokenization {
  type: string,
  leftSpace: (string) => boolean,
  newLine: (string) => boolean,
  cleanup: (string) => string
}

export const available_tokenizations = {
  'gpt': {
    type: 'gpt',
    leftSpace: (a) => a.startsWith('Ġ'),
    newLine: (a) => a.startsWith('Ċ'),
    cleanup: token => {
      token = (token.startsWith('Ġ')) ? token.slice(1) : ((token.startsWith('Ċ') || token.startsWith('â')) ? " " : token);
      // token = (token.startsWith('â')) ? '–' : token;
      // token = (token.startsWith('ľ')) ? '“' : token;
      // token = (token.startsWith('Ŀ')) ? '”' : token;
      // token = (token.startsWith('Ļ')) ? "'" : token;
      return token_encoding(token)
    }
  },
  'bert': {
    type: 'bert',
    leftSpace: (a) => !a.startsWith('##'),
    newLine: (a) => false,
    cleanup: token => {
      token = (token.startsWith('##')) ? token.slice(2) : token;
      return token_encoding(token)
    }
  }

} as { [key: string]: Tokenization }

function token_encoding(token) {

  try {
    token = decodeURIComponent(escape(token));
  } catch {
    console.log(token, '-- token is hard');
  }
  return token;
}
