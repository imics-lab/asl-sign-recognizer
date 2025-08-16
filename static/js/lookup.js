//--------DOM Elements--------
//--Buttons--
const submitButton = document.getElementById('submit-button');
const playButton = document.getElementById('play-button');
const restartButton = document.getElementById('restart-button');

//--Video Box--
const videoPlayer = document.getElementById('videoDisplay');

//--Titles--
const searchedWordTitle = document.getElementById('searched-word-title');

//--TextBox--
const input = document.getElementById('search-box')

//--Text
const errorMessage = document.getElementById("error-message");

function updateButtonStates() {
    /* Update Button States */
    if (videoPlayer.ended) {
        // After finishing
        playButton.hidden = true;
        if (errorMessage.hidden === true){
        restartButton.hidden = false;
        } else {
            restartButton.hidden = true;
        }
    } else if (!videoPlayer.paused) {
        // While playing
        playButton.hidden = true;
        restartButton.hidden = true;
    } else if (videoPlayer.paused && videoPlayer.currentTime === 0) {
        // Before playing for the first time
        playButton.hidden = false;
        restartButton.hidden = true;
    } else {
        // Paused mid-playback
        playButton.hidden = false;
        restartButton.hidden = true;
    } 
}

function loadSearchedVideo() {
    /* Loads video when user searches word through textbox */
    const searchWord = document.getElementById('search-box').value.toLowerCase();
    errorMessage.hidden = true;

    fetch("/get_video_file_name", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: searchWord })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(errData => {
                throw new Error(errData.error || `HTTP error! status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        // If word exists
        searchedWordTitle.textContent = searchWord.charAt(0).toUpperCase() + searchWord.slice(1);
        searchedWordTitle.hidden = false;
        videoPlayer.hidden = false;
        videoPlayer.src = `/static/videos/${data.videoFile}`;
        updateButtonStates();
    })
    .catch(error => {
        console.error("Error fetching video:", error.message);

        // Show error message to user and ensures other elements are not shown
        errorMessage.textContent = error.message;
        playButton.hidden = true;
        searchedWordTitle.hidden = true;
        errorMessage.hidden = false;
        videoPlayer.hidden = true;
    });
}

//--Event Listeners--
submitButton.addEventListener("click", loadSearchedVideo);

// Plays video and updates UI
playButton.addEventListener("click", () => {
    videoPlayer.play();
    updateButtonStates();
});

// Uppdates buttons(UI) when video is done playing
videoPlayer.addEventListener("ended", updateButtonStates);

// Plays back video
restartButton.addEventListener("click", () => {
    videoPlayer.play();
    updateButtonStates();
});

// Lets user press enter on keyboard when input box is selected
input.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
   loadSearchedVideo();
  }
});