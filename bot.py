import discord
import os
from discord.ext import commands
from dotenv import load_dotenv
import data_parse

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents,strip_after_prefix=True)

@bot.command(description="Sends the bot's latency.")
async def ping(ctx):
    await ctx.send(f"Pong! Latency is {round(bot.latency * 1000)} ms")

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

    try:
        await data_parse.getMessage(message)
    except Exception as e:
        print(f"Error processing message: {e}")
        await message.channel.send(f'an error occurred: {e}')


if __name__ == '__main__':
    token = os.getenv('DISCORD_TOKEN')
    if token:
        bot.run(token)
    else:
        print("Error: DISCORD_TOKEN not found in .env file.")
