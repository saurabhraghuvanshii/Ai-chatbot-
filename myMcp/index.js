import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
    name: 'Weather check',
    version: '1.0.0',
})

async function getWeatherByCity(city = '') {
    if (city.toLowerCase() === 'varanasi') {
        return { temp: '34C', forcast: 'chances of high hottness' }
    }

    if (city.toLowerCase() === 'noida') {
        return { temp: '38C', forcast: 'chances of rain' }
    }

    return { temp: null, error: 'Unable to get data' }
}

server.tool('getWeatherDataByCityName', {
    city: z.string(),
}, async ({ city }) => {
    return { content: [{ type: "text", text: JSON.stringify(await getWeatherByCity(city)) }] }
})

async function init() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}

init();
