// book/_addons/language_switcher.js
function switchLanguage(newLang) {
    var currentPath = window.location.origin + window.location.pathname;
    var supportedLangs = ['en', 'ko']; // Update this array to match your LANGUAGES array
    var langRegex = new RegExp('\\b(' + supportedLangs.join('|') + ')\\b');

    if (langRegex.test(currentPath)) {
        // If the current path contains a language code, replace it
        var newPath = currentPath.replace(langRegex, newLang);
        window.location.href = newPath + window.location.search + window.location.hash;
    }
}
