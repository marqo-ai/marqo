import moviepy
import PIL

def convert_to_listofPIL(video):
    frames = list(video.iter_frames())
    return [PIL.Image.fromarray(i.astype("uint8"), 'RGB') for i in frames]

def chunk_video(field_content : str, device = "cpu", method = "simple"):
    clip = field_content
    duration = clip.duration
    content_chunks = []
    reminder = duration % 10

    if duration < 15:
        num_of_chunks = 1
        content_chunks.append((num_of_chunks - 1) * 10, num_of_chunks * 10 + reminder)
    else:
        if reminder > 5.0:
            num_of_chunks = duration // 10 + 1
        else:
            num_of_chunks = duration // 10
        for i in range(int(num_of_chunks) - 1):
            content_chunks.append(clip.subclip(i * 10, i * 10 + 10))
        content_chunks.append(clip.subclip(10 * (num_of_chunks - 1), 10 * num_of_chunks + reminder))

    trunk_duration = [i.duration for i in content_chunks]
    assert duration == sum(trunk_duration), "The truncks are not correct"
    return content_chunks, trunk_duration
