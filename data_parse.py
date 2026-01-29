async def getMessage(message):
    print(f'Message from {message.author}: {message.content}')

    await message.channel.send(f'Received your message: {message.content}')
    return True