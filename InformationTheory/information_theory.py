"""
Information Theory Lab

Danny Perkins
MATH 403 (001)
9/8/24
"""

import numpy as np
import wordle

# Problem 1
def get_guess_result(guess, true_word):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the secret word is "boxed" and the provided guess is "excel", the
    function should return [0,1,0,2,0].

    Arguments:
        guess (string) - the guess being made
        true_word (string) - the secret word
    Returns:
        result (list of integers) - the result of the guess, as described above
    """
    # Make sure the input makes sense
    assert len(guess) == 5, "The guess does not have five letters"
    assert len(true_word) == 5, "The true word does not have five letters"
        
    output = [-1]*5   # Initialize all spots as -1
    for i in range(5):
        if(guess[i] == true_word[i]): output[i] = 2
        elif(guess[i] not in true_word): output[i] = 0
        else:
            count = true_word.count(guess[i])
            for j in range(5):  # Fill green spots first
                if(guess[i] == guess[j] and guess[j] == true_word[j]):
                    output[j] = 2
                    count -= 1
            for k in range(5):  # Remove 1 from count for each time this letter was already yellow
                if(guess[i] == guess[k] and output[k] == 1): count -= 1
            if count > 0: output[i] = 1   
            else: output[i] = 0 # Already used up all spots
    
    return output


# Helper function
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words

# Problem 2
def compute_highest_entropy(all_guess_results, allowed_guesses):
    """
    Compute the entropy of each allowed guess.

    Arguments:
        all_guess_results ((n,m) ndarray) - the array found in
            all_guess_results.npy, containing the results of each
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_guesses (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
    """
    n = len(allowed_guesses)
    entropies = []
    for i in range(n):
        guess_results = all_guess_results[i]
        unique_values = np.unique(guess_results, return_counts=True)[1] # Find number of unique occurences
        probabilities = unique_values / np.sum(unique_values)  # Get probabilities
        entropy = np.sum(np.log2(probabilities) * -probabilities)  # Calculate entropy
        entropies.append(entropy)
    return allowed_guesses[np.argmax(entropies)]

# Problem 3
def filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already have an array of the result of all guesses for all possible words,
    we will use this array instead of recomputing the results.

    Return a filtered list of possible words that are still possible after
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words
    still possible after making the guess. This array will be used to compute
    the entropies for making the next guess.

    Arguments:
        all_guess_results (2-D ndarray)
            The array found in all_guess_results.npy,
            containing the result of making any allowed
            guess for any possible secret word
        allowed_guesses (list of str)
            The list of words we are allowed to guess
        possible_secret_words (list of str)
            The list of possible secret words
        guess (str)
            The guess we made
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (2-D ndarray) The filtered array of guess results
    """
    # Convert data to right type
    possible_secret_words = np.array(possible_secret_words)
    
    word_length = len(guess) # Should always be 5
    assert word_length == 5, "Guess does not contain 5 letters"
    
    # Convert the actual result to correct format
    float_result = 0
    for i in range(5):  # Convert result to the right format
        float_result += result[i] * 3**i
    
    # Find all results for the guess
    guess_index = allowed_guesses.index(guess)
    guess_results = all_guess_results[guess_index]

    mask = (guess_results == float_result)  # Get indices of of words that yield the same result
    possible_secret_words = possible_secret_words[mask]
    all_guess_results = all_guess_results[:, mask]
    
    return possible_secret_words, all_guess_results
    
    

# Problem 4
def play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.

    Return how many guesses were used.

    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy,
            containing the result of making any allowed
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses

        word (optional)
            If not None, this is the secret word; can be used for testing.
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    while True:
        if len(possible_secret_words) == 1:  # Choose the only possible word if onw is left
            guess = possible_secret_words[0]
        else:
            guess = np.random.choice(allowed_guesses)  # Choose randomly from valid words
        result, num_guesses = game.make_guess(guess)  # Make the guess
        if game.is_finished(): break
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)
        
    return num_guesses

    

# Problem 5
def play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.

    Return how many guesses were used.

    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy,
            containing the result of making any allowed
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses

        word (optional)
            If not None, this is the secret word; can be used for testing.
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    
    while True:
        if len(possible_secret_words) == 1:  # Choose the only possible word if onw is left
            guess = possible_secret_words[0]
        else:
            guess = compute_highest_entropy(all_guess_results, allowed_guesses)  # Choose word with highest entropy
        result, num_guesses = game.make_guess(guess)  # Make the guess
        if game.is_finished(): break
        possible_secret_words, all_guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)
        
    return num_guesses

# Problem 6
def compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.


    Arguments:
        all_guess_results ((n,m) ndarray)
            The array found in all_guess_results.npy,
            containing the result of making any allowed
            guess for any possible secret word
        possible_secret_words (list of str)
            list of possible secret words
        allowed_guesses (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    naive_guesses = 0
    entropy_guesses = 0
    for i in range(n):
        game = wordle.WordleGame()  # Create game
        naive_guesses += play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False)
        game = wordle.WordleGame()  # Reset game
        entropy_guesses += play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=False)
        
    return naive_guesses / n, entropy_guesses / n  # Return averages
        


if __name__=="__main__":
    # Prob 1
    # print(get_guess_result("excel", "boxed"))
    # print(get_guess_result("stare", "train"))
    # print(get_guess_result("green", "pages"))
    # print(get_guess_result("abate", "vials"))
    # print(get_guess_result("robot", "older"))
    
    # Prob 2
    # all_guess_results = np.load("all_guess_results.npy")
    # allowed_guesses = load_words("allowed_guesses.txt")
    # print(compute_highest_entropy(all_guess_results, allowed_guesses))
    
    # Prob 3
    # all_guess_results = np.load("all_guess_results.npy")
    # allowed_guesses = load_words("allowed_guesses.txt")
    # possible_secret_words = load_words("possible_secret_words.txt")
    # guess = "boxes"
    # result = (0, 0 ,0, 2, 1)
    # possible_secret_words, guess_results = filter_words(all_guess_results, allowed_guesses, possible_secret_words, guess, result)
    # print(len(possible_secret_words), possible_secret_words)
    # print(np.shape(guess_results), guess_results)
    
    # Prob 4
    # game = wordle.WordleGame()
    # all_guess_results = np.load("all_guess_results.npy")
    # allowed_guesses = load_words("allowed_guesses.txt")
    # possible_secret_words = load_words("possible_secret_words.txt")
    # num_guesses = play_game_naive(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=True)
    # print(num_guesses)
    
    # Prob 5
    # game = wordle.WordleGame()
    # all_guess_results = np.load("all_guess_results.npy")
    # allowed_guesses = load_words("allowed_guesses.txt")
    # possible_secret_words = load_words("possible_secret_words.txt")
    # print(play_game_entropy(game, all_guess_results, possible_secret_words, allowed_guesses, word=None, display=True))
    
    # Prob 6
    all_guess_results = np.load("all_guess_results.npy")
    allowed_guesses = load_words("allowed_guesses.txt")
    possible_secret_words = load_words("possible_secret_words.txt")
    naive_guesses, entropy_guesses = compare_algorithms(all_guess_results, possible_secret_words, allowed_guesses, n=20)
    print("Naive method:", naive_guesses)
    print("Entropy method:", entropy_guesses)
    
    pass