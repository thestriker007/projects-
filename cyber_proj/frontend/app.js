const TARGET_PHRASE = ".tie5Roanl";

// Specific sequence of keys expected by the DSL dataset
const KEY_SEQUENCE = ['.', 't', 'i', 'e', '5', 'Shift', 'o', 'a', 'n', 'l', 'Enter'];

// State
let keyTimings = {};
let hasError = false;

// DOM Elements
const statusContainer = document.getElementById('status-container');
const statusMessage = document.getElementById('status-message');
const usernameInput = document.getElementById('username');
const passphraseInput = document.getElementById('passphrase-input');
const actionBtn = document.getElementById('action-btn');

passphraseInput.addEventListener('keydown', (e) => {
    // Reset if input is empty
    if (passphraseInput.value === '') {
        keyTimings = {};
        hasError = false;
        hideStatus();
    }

    // No backspaces allowed for pure timing dataset
    if (e.key === 'Backspace' || e.key === 'Delete') {
        hasError = true;
        showStatus('Errors are not allowed. Please clear the input and try again from the start.', 'error');
        return;
    }

    // Record down time for valid key
    const key = e.key;
    // We only care about the first time a key goes down to handle auto-repeat
    if (!keyTimings[key]) {
        keyTimings[key] = { down: Date.now(), up: null };
    }

    // If enter is pressed, attempt login
    if (key === 'Enter') {
        e.preventDefault();
        if (hasError) return;
        
        if (passphraseInput.value !== TARGET_PHRASE) {
            showStatus('Incorrect passphrase. Please clear and try again.', 'error');
            hasError = true;
            return;
        }

        processLogin();
    }
});

passphraseInput.addEventListener('keyup', (e) => {
    const key = e.key;
    if (keyTimings[key] && keyTimings[key].up === null) {
        keyTimings[key].up = Date.now();
    }
});

usernameInput.addEventListener('input', validateForm);
passphraseInput.addEventListener('input', validateForm);

function validateForm() {
    const user = usernameInput.value.trim();
    const phrase = passphraseInput.value;
    actionBtn.disabled = !(user.length > 0 && phrase === TARGET_PHRASE && !hasError);
}

function processLogin() {
    const user = usernameInput.value.trim();
    if (!user) {
        showStatus('Please enter a Subject Identity (e.g., s002).', 'error');
        return;
    }

    try {
        const features = extract31Features();
        authenticateSubject(user, features);
    } catch (err) {
        showStatus('Failed to extract timing features. Try typing again naturally.', 'error');
        console.error(err);
    }
}

function extract31Features() {
    // Helper to get times in SECONDS to match DSL dataset format
    const H = (k) => (keyTimings[k].up - keyTimings[k].down) / 1000.0;
    const DD = (k1, k2) => (keyTimings[k2].down - keyTimings[k1].down) / 1000.0;
    const UD = (k1, k2) => (keyTimings[k2].down - keyTimings[k1].up) / 1000.0;

    // Check if all necessary keys were recorded
    for (let k of KEY_SEQUENCE) {
        if (!keyTimings[k] || keyTimings[k].up === null) {
            throw new Error(`Incomplete timing for key: ${k}`);
        }
    }

    const features = [
        H('.'), DD('.', 't'), UD('.', 't'),
        H('t'), DD('t', 'i'), UD('t', 'i'),
        H('i'), DD('i', 'e'), UD('i', 'e'),
        H('e'), DD('e', '5'), UD('e', '5'),
        H('5'), DD('5', 'Shift'), UD('5', 'Shift'),
        H('Shift'), DD('Shift', 'o'), UD('Shift', 'o'),
        H('o'), DD('o', 'a'), UD('o', 'a'),
        H('a'), DD('a', 'n'), UD('a', 'n'),
        H('n'), DD('n', 'l'), UD('n', 'l'),
        H('l'), DD('l', 'Enter'), UD('l', 'Enter'),
        H('Enter')
    ];

    return features;
}

actionBtn.addEventListener('click', () => {
    if (!actionBtn.disabled) {
        processLogin();
    }
});

async function authenticateSubject(username, featureVector) {
    actionBtn.innerText = 'Verifying Subject Sequence...';
    actionBtn.disabled = true;
    passphraseInput.disabled = true;
    
    try {
        const response = await fetch('http://127.0.0.1:5000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: username,
                sample: featureVector
            })
        });
        
        const data = await response.json();
        if (response.ok && data.authenticated) {
            showStatus(data.message, 'success');
        } else {
            showStatus(data.error || data.message || 'Access Denied.', 'error');
        }
    } catch (err) {
        showStatus('Backend ML server unreachable. Is it running?', 'error');
    } finally {
        actionBtn.innerText = 'Login & Authenticate';
        actionBtn.disabled = false;
        passphraseInput.disabled = false;
        passphraseInput.value = '';
        passphraseInput.focus();
        keyTimings = {};
        hasError = false;
    }
}

// UI Helpers
function showStatus(msg, type) {
    statusMessage.innerText = msg;
    statusContainer.className = `status-container ${type}`;
}

function hideStatus() {
    statusContainer.className = 'status-container hidden';
}
