import moviepy
import PIL
from numpy import ndarray
from marqo.s2_inference.types import VideoType


def convert_to_listofPIL(video: ndarray) -> VideoType:
    frames = list(video.iter_frames())
    return [PIL.Image.fromarray(i.astype("uint8"), 'RGB') for i in frames]


def chunk_video(field_content: str, trunk_length = 10, method="merge", device = "cpu"):
    '''
    The length of each chunk is 10s as the x_clip model is trained by videos with 10 seconds.


    The last chunk is very likely to be less than 10s, we solve the problem using the following rules:
    If the last chunk is longer than 5 seconds, we keep the chunk.
    If the last chunk is less than 5 seconds, we add this chunk to the previous one.

    For example, for a video with 34s, we will have 3 chunks with length [10s, 10s, 14s]
    As for a video with 38s, we will have 4 chunks with length [10s, 10s, 10s, 8s]
    '''

    clip = field_content
    duration = clip.duration
    content_chunks = []
    reminder = duration % trunk_length

    if method == "merge":
        if duration < int(trunk_length * 1.5):
            num_of_chunks = 1
            content_chunks.append((num_of_chunks - 1) * trunk_length, num_of_chunks * trunk_length + reminder)
        else:
            if reminder > (0.5 * trunk_length):
                num_of_chunks = duration // trunk_length + 1
            else:
                num_of_chunks = duration // trunk_length

            for i in range(int(num_of_chunks) - 1):
                content_chunks.append(clip.subclip(i * trunk_length, (i + 1) * trunk_length))

            # Merge the reminder to the last chunk
            content_chunks.append(clip.subclip(trunk_length * (num_of_chunks - 1), duration))

    elif method == "simple":
        num_of_chunks = duration // trunk_length + (reminder > 0)

        for i in range(int(num_of_chunks) - 1):
            content_chunks.append(clip.subclip(i * trunk_length, (i + 1) * trunk_length))
        content_chunks.append(clip.subclip(trunk_length * (num_of_chunks - 1), duration))

    else:
        raise RuntimeError("Unsupported truncation method!")

    trunk_duration = [i.duration for i in content_chunks]
    if duration != sum(trunk_duration):
        raise RuntimeError("The trunks are not correct!")
    return content_chunks, trunk_duration
