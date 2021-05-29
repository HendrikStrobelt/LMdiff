/** Convert an object into a URL parameter string for GET requests
 *
 * @param base Base URL atop which to add GET parameters
 * @param params Object to insert into a URL string
 * @returns GET query string
 */
export function makeUrl(base: string, params?: object):string {
    if (params){
        let out: string = base + "?";

        Object.keys(params).forEach( k => {
            out += encodeURIComponent(k);
            out += '=';
            out += encodeURIComponent(params[k]);
            out += "&";
        })
        return out.replace(/&$/g, "");
    }
    else {
        return base;
    }
};

/** Convert object information the message for a POST request
 *
 * @param toSend Simple object to put into a POST request
 * @returns Object with appropriate headers to send as a POST
 */
export const toPayload = (toSend) => {return {
    method:"POST",
    body:JSON.stringify(toSend),
    headers: {
            "Content-type": "application/json; charset=UTF-8"
        }
}}
