"""R-precision có tính điểm nghệ sĩ, theo độ đo chính của RecSys Challenge 2018 (cũng dùng MPD)."""


def r_precision(recommended, hidden, artist_of=None, artist_weight=0.25):
    """Chấm R = |hidden| gợi ý đầu, chia cho R.

    Gợi ý trúng bài bị giấu được 1 điểm. Gợi ý không trúng bài nhưng cùng nghệ sĩ với
    một bài bị giấu được artist_weight điểm; mỗi nghệ sĩ chỉ được tính một lần, và
    nghệ sĩ đã có bài trúng thì không được cộng thêm, nên điểm tối đa vẫn là 1.
    artist_of(item) trả về nghệ sĩ của bài; None thì chỉ tính trúng bài.
    """
    if not hidden:
        raise ValueError("hidden rỗng: playlist này không đánh giá được")
    hidden = set(hidden)
    top = recommended[:len(hidden)]
    hits = [item for item in top if item in hidden]
    score = float(len(hits))
    if artist_of is not None:
        hidden_artists = {artist_of(item) for item in hidden}
        credited = {artist_of(item) for item in hits}
        for item in top:
            if item not in hidden:
                artist = artist_of(item)
                if artist in hidden_artists and artist not in credited:
                    credited.add(artist)
                    score += artist_weight
    return score / len(hidden)
