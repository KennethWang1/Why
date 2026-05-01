Wow, training my own AI chatbot!! What could possibly go wrong?

# Description

A 22 million param model (1GB-ish of RAM) trained on both publicly available data as well as privately obtained ones. Runs on a Raspberry Pi currently and responds to messages sent in the bot channel of whatever server it is in. Note: It does hallucinate a lot and is not very good. Please use proper judgement as I am not responsible for the actions anyone takes as a result of something the LLM said.

# Purpose
For me to learn how to create and train an AI model from scratch and see what limitations come up from it. Also, because it is funny to see a chatbot go insane!

# Usage

Simply message in the bot channel of a server with the bot. It should respond (albeit slowly) with 2 responses. Please select the one you prefer (this is for future training purposes!) Should a response not come, either wait (as responses can take up to 30 minutes), or message me informing the bot is down (you can see the status through Discord).

### Bot commands
#### Public
!acc - checks current bot accuracy (defaults to 0 if no training cycle has been run after a restart)
!ping - returns pong as well as the current ping
!save - depreciated (used to save vocabulary of the tokenizer, but is now automated)

#### Private
!start - starts model training
!stop - stops model training after the next cycle

# Details
An LLM with a simple transformer model, an off-the-shelf tokenizer, and an off-the-shelf RAG system to provide it the best chance when finding the optimal response. Really nothing extraordinary. This is all integrated with Discord's API to create a custom bot!

# licensing
Should (for some reason) you want to use this AI, please reach out before doing so. Also, I am not responsible for anything that comes from the code or what the LLM says.
