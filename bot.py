import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
import data_parse
import runner

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
            response = await bot.loop.run_in_executor(None, runner.generate_response, input_text)
            
            if response.strip():
                await message.channel.send(response)
            else:
                await message.channel.send("...")
        except Exception as e:
            print(f"Error generating response: {e}")
            await message.channel.send(f"Error: {e}")

    try:
        await data_parse.getMessage(message)
    except Exception as e:
        print(f"Error processing message: {e}")
        await message.channel.send(f'an error occurred: {e}')


if __name__ == '__main__':
    data_parse.load_vocab()
    token = os.getenv('DISCORD_TOKEN')
    if token:
        try:
            bot.run(token)
        finally:
            data_parse.save_vocab()
    else:
        print("Error: DISCORD_TOKEN not found in .env file.")