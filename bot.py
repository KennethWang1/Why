import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
import data_parse
import runner
import tester

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents,strip_after_prefix=True)

@bot.command(description="Start the training loop.")
async def start(ctx):
    if runner.training_active:
        await ctx.send("Training is already in progress.")
    else:
        runner.start_training()
        await ctx.send("Training started in background.")

@bot.command(description="Stop the training loop.")
async def stop(ctx):
    if not runner.training_active:
        await ctx.send("Training is not currently running.")
    else:
        runner.stop_training()
        await ctx.send("Stopping training... (waiting for current cycle to finish, model will be saved)")

@bot.command(aliases=['acc'], description="Get the current model accuracy.")
async def accuracy(ctx):
    acc = runner.get_current_accuracy()
    await ctx.send(f"Current Model Accuracy: {acc:.4f}")

@bot.command(description="Sends the bot's latency.")
async def ping(ctx):
    await ctx.send(f"Pong! Latency is {round(bot.latency * 1000)} ms")

@bot.command(description="saves the vocabulary to a file.")
async def save(ctx):
    data_parse.save_vocab()
    await ctx.send("Vocabulary saved to file.")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if message.channel.name != "bot":
        return
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return
    
    if message.channel.name == "bot":

        input_text = f"{message.author.name}: {message.content}"
        
        try:
            response1 = await bot.loop.run_in_executor(
                None,
                lambda: runner.generate_response(input_text, temperature=0.85, top_k=30, min_tokens=4),
            )
            response2 = await bot.loop.run_in_executor(
                None,
                lambda: runner.generate_response(input_text, temperature=1.05, top_k=60, min_tokens=4),
            )
            data_parse.add_embeddings(input_text)
            data_parse.add_embeddings(response1)
            message_content = f"1.\n{response1}\n2.\n{response2}"
            message_content.strip()
            bot_message = await message.channel.send(message_content)
            await bot_message.add_reaction("1️⃣")
            await bot_message.add_reaction("2️⃣")

            tester.add_response(input_text, response1, response2)
        except Exception as e:
            print(f"Error generating response: {e}")
            await message.channel.send(f"Error: {e}")

@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user:
        return

    # Only process votes on response messages sent by this bot.
    if reaction.message.author != bot.user:
        return

    if reaction.emoji not in ["1️⃣", "2️⃣"]:
        return

    selected = reaction.message.content.split("\n2.\n")[0].replace("1.\n", "").strip()
    if reaction.emoji == "1️⃣":
        #print(f"Validating response: {selected} with choice 1")
        tester.validate_response(selected, 1)
    else:
        #print(f"Validating response: {selected} with choice 2")
        tester.validate_response(selected, 2)
    #print(f"Reaction added: {reaction.emoji} by {user.name}")
    #print(f"message content: {selected}")

if __name__ == '__main__':
    data_parse.load_vocab()
    token = os.getenv('DISCORD_TOKEN')
    if token:
        try:
            bot.run(token)
        except Exception as e:
            print(f"Error running bot: {e}")
    else:
        print("Error: DISCORD_TOKEN not found in .env file.")