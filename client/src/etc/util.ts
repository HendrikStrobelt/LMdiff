import * as d3 from "d3";

export type D3Sel = d3.Selection<any, any, any, any>

let the_unique_id_counter = 0;
export function simpleUId({prefix = ''}): string {
    the_unique_id_counter += 1;
    return prefix + the_unique_id_counter;
}


export class Transforms {
    static translate(x, y):string {
        return `translate(${x},${y})`
    }

    static rotate(deg):string {
        return `rotate(${deg})`
    }
}

export function token_cleanup(token) {

    token = (token.startsWith('Ġ')) ? token.slice(1) : ((token.startsWith('Ċ') || token.startsWith('â')) ? " " : token);
    // token = (token.startsWith('â')) ? '–' : token;
    // token = (token.startsWith('ľ')) ? '“' : token;
    // token = (token.startsWith('Ŀ')) ? '”' : token;
    // token = (token.startsWith('Ļ')) ? "'" : token;

    try {
        token = decodeURIComponent(escape(token));
    } catch{
        console.log(token, '-- token is hard');
    }
    return token;
}

/**
 * From https://stackoverflow.com/questions/33855641/copy-output-of-a-javascript-variable-to-the-clipboard
 *
 * Copy specified `text` to clipboard
 */
export function copyToClipboard(text:string) {
    var dummy = document.createElement("textarea");
    // to avoid breaking orgain page when copying more words
    // cant copy when adding below this code
    document.body.appendChild(dummy);
    //Be careful if you use textarea. setAttribute('value', value), which works with "input" does not work with "textarea". – Eduard
    d3.select(dummy).attr('value', text);
    dummy.value = text
    dummy.select();
    console.log(dummy);
    document.execCommand("copy");
    document.body.removeChild(dummy);
}
