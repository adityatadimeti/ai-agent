import logging

logger = logging.getLogger("discord")

async def find_existing_thread(channel, thread_name):
    """Find an existing thread by name, with timeout protection"""
    try:
        async for thread in channel.archived_threads(limit=100):
            if thread.name == thread_name:
                return thread
                
        for thread in channel.threads:
            if thread.name == thread_name:
                return thread
                
        return None
    except Exception as e:
        logger.error(f"Error finding thread: {str(e)}")
        return None

async def queue_message(outbound_message_queue, channel, content):
    """Queue a message to be sent with rate limiting"""
    logger.info(f"Queueing message: {content}")
    await outbound_message_queue.put((channel, content))

async def queue_chunked_message(outbound_message_queue, message, response):
    """Queue chunked messages with rate limiting"""
    chunks = split_response_into_chunks(response)
    
    i = 0
    for chunk in chunks:
        print(i)
        i += 1
        logger.info(f"Queueing chunk: {chunk}")
        await outbound_message_queue.put((message.channel, chunk))

def split_response_into_chunks(response, max_length=1900):
    """Split response into chunks, respecting markdown and length limits"""
    chunks = []
    sections = response.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in sections:
        # If line starts with heading (#) and we have content, start new chunk
        if line.lstrip().startswith('#'):
            # Check if it's a level 4 heading and convert to level 2
            if line.lstrip().startswith('####'):
                line = line.replace('####', '##')
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
        # Add line to current chunk
        line_length = len(line) + 1  # +1 for newline
        if current_length + line_length > max_length:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(line)
        current_length += line_length
    
    # Add any remaining content as final chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        
    return chunks