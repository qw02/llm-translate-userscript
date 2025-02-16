# Web Novel Translator Userscript

This userscript provides a powerful, in-browser solution for translating web novels from Japanese to English, using a Retrieval-Augmented Generation (RAG) pipeline powered by Large Language Models (LLMs).  It's designed to work directly on web novel websites, offering a seamless translation experience.

## Features

*   **Automatic Translation:** Translate Japanese web novel pages directly in your browser with a single click.
*   **RAG Pipeline:** Employs a sophisticated RAG (Retrieval-Augmented Generation) system. This means it builds a dynamic glossary of terms (names, places, special items, etc.) from the novel itself. This glossary is used to improve the consistency and accuracy of the translation, especially for recurring elements.
*   **Multi-Provider Support:** Supports multiple LLM providers:
    *   **OpenRouter:**  Allows you to use a wide variety of models from different providers (e.g., Google's Gemini, Anthropic's Claude, OpenAI's GPT-4o, and many more).
    *   **Anthropic:** Directly use Anthropic's Claude models.
    *   **OpenAI:** Directly utilizes GPT-4o / GPT-4o-mini.
*   **Model Selection:**  Offers a selection of LLM models for each stage of the translation process (glossary generation, glossary updating, and the main translation).  You can choose models based on your needs and preferences.
*   **Persistent Glossary:**  The generated glossary is saved and reused across chapters of the same series, ensuring consistent translation of key terms throughout the novel.
*   **In-Browser Glossary Editor:**  Provides an in-browser editor to manually view, edit, and refine the glossary.  This gives you fine-grained control over the translation of specific terms.

## How to Use

1.  **Install a Userscript Manager:** You'll need a userscript manager like Tampermonkey or Violentmonkey.
2.  **Install the Script:** Install the userscript by copying the code from this repository and adding a new user script using the manager.
3.  **Navigate to a Supported Novel Page:** Go to a chapter page on one of the supported websites (syosetu.com or kakuyomu.jp).
4.  **Configure API Keys:** Click the "Manage API Keys" button that appears on the page.  Enter your API keys for the LLM providers you want to use (OpenRouter, Anthropic, OpenAI).  *Note: You only need keys for the providers you intend to use.*
5.  **Select a Provider and Models:** Choose your preferred LLM provider and the desired models for each stage of the translation process (glossary generation, updating, and translation).
6.  **Translate:** Click the "Translate" button. The script will begin translating the current chapter, replacing the Japanese text with English.
7.  **Optional: Edit Glossary:** Click "Edit Metadata" to view and edit the glossary.  This is useful for fine-tuning the translation of specific terms.

## Important Notes

*   **API Keys:** You'll need to obtain API keys from the LLM providers you want to use.  This usually involves creating an account and potentially subscribing to a plan.
*   **Cost:** Using LLMs can incur costs, depending on the provider and the models you choose.  Be aware of the pricing structure of your chosen provider.
*   **Translation Quality:** While the RAG pipeline and LLMs significantly improve translation quality, especially for consistent terminology, it's not perfect.  The translation may still require some manual review and editing, particularly for nuanced language or complex sentences.
