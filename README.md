# LLM Web Novel Translator

Userscript that translates web novels using a multi-stage RAG pipeline using LLMs, designed to generate high-quality, translations for long-form content like entire novel series. Uses an automatically generated and updated internal glossary for consistency of translated names, terms, and proper nouns.

Core pipeline:
1.  **Generates new glossary entries:** Scans the chapter to build a persistent, series-specific dictionary for characters, locations, and terms.
2.  **Updates the existing glossary:** Uses an agentic workflow to intelligently merge new terms and de-duplicate existing ones.
3.  **Text chunking / segmentation:** Employs an LLM to semantically segment the chapter into coherent blocks of around 5 sentences or 200 chars.
4.  **Translates:** Translates each chunk using the glossary as context, ensuring consistency.

## Features

-   **Consistent Translation:** Uses a persistent glossary (RAG) to ensure names and terminology are translated consistently across hundreds of chapters.
-   **Multi-Provider Support:** Integrates with any OpenAI-compatible API, with built-in support for OpenRouter, Anthropic, DeepSeek, OpenAI, Google Gemini, xAI and NanoGPT.
-   **Agentic Glossary Management:** Autonomously maintains and refines the translation glossary when script is ran on new chapters.
-   **Configurable Pipeline:** Fine-tune every stage of the process by selecting different models for glossary generation, chunking, and final translation.

## Installation

1.  Install a userscript manager extension for your browser, such as [Tampermonkey](https://www.tampermonkey.net/) or Greasemonkey.
2.  Download the script [`webpage-translator-2.0.user.js`](https://github.com/qw02/llm-translate-userscript/raw/main/webpage-translator-2.0.user.js) and load it in the userscript manager.

## Setup & Usage

This script uses your own API keys (BYOK).

1.  **Set API Keys:** After installing, open a supported web novel page. A control panel will appear. Click the **`API Keys`** button and enter your key for at least one provider.
2.  **Configure Pipeline:**
    -   Open the **`Models`** section in the control panel.
    -   Assign an LLM from an active provider to each stage of the pipeline.
    -   To skip glossary generation on a per-translation basis, check "Skip Glossary Generation".
3.  **(Optional) Adjust Options:** Open the **`Options`** section to control translation style (e.g., narrative voice, name order, honorifics). Custom instructions can also be added.
4.  **Translate:** Click **`Start Translation`**.

## For Developers: Extending the Script

The script is designed to be easily extensible.

### Supporting a New Website

To add support for a new web novel site, add an entry to the `DOMAIN_CONFIGS` object. You need to implement two functions:

-   `getSeriesId`: Extracts a unique ID for the novel series from the URL, used for the glossary key.
-   `extractText`: Scrapes the page and returns a `Map` where keys are the paragraph element IDs and values are the raw text content.

```javascript
// Example for a new site
'new-site.com': {
  domainCode: '3', // A unique numeric code for the domain
  getSeriesId: (url) => {
    // Logic to parse the URL and return a series identifier
    const match = url.match(/\/series\/(\d+)/);
    return match ? match[1] : null;
  },
  extractText: () => {
    // Logic to find and process all paragraph elements
    const paragraphMap = new Map();
    const pElements = document.querySelectorAll('.novel-content > p');
    pElements.forEach(p => {
      if (p.id) {
        // It's recommended to preprocess text here (e.g., handle ruby tags)
        paragraphMap.set(p.id, p.textContent);
      }
    });
    return paragraphMap;
  },
},
```

### Adding a New Model

To add a new model from an existing provider, find the provider's entry in `PROVIDER_CONFIGS` and add a new model object to the `models` array. The `id` must be unique across all providers.

```javascript
// Example for adding a new OpenAI model
openai: {
  models: [
    // ... existing models
    { id: '3-5', model: 'gpt-6-turbo', label: 'GPT-6 Turbo (New)' },
  ],
  limits: { /* ... define which stages this model can be used for ... */ }
}
```

### Adding a New API Provider

Adding a new provider requires creating an API adapter. Most modern providers use an OpenAI-compatible API, making this straightforward.

1.  Add the provider's key and endpoint to `PROVIDER_API_CONFIG`.
2.  Add a model list for the new provider in `PROVIDER_CONFIGS`.
3.  Add an entry in `ApiAdapters`. For most cases, you can reuse `createOpenAIApiAdapter`.

```javascript
// For an OpenAI-compatible provider
const ApiAdapters = {
  // ... existing adapters
  newprovider: createOpenAIApiAdapter(),
};

// For a provider with a unique request/response structure
const ApiAdapters = {
  // ... existing adapters
  customprovider: {
    buildRequest(endpoint, modelId, messages, modelConfig, apiKey) {
      // Return an object with { url, headers, payload }
    },
    parseResponse(response) {
      // Parse the raw response and return { completion, reasoning }
    }
  }
};
```
