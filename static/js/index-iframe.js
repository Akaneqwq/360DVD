document.addEventListener('DOMContentLoaded', function () {
    const observerCallback = function (entries, observer) {
        entries.forEach(entry => {
            const iframe = entry.target;
            if (entry.isIntersecting) {
                iframe.src = iframe.getAttribute('data-src');
            } else {
                iframe.src = "";
            }
        });
    };
    let observer = new IntersectionObserver(observerCallback, {
        threshold: 0.2
    });
    for (const iframe of document.getElementsByTagName("iframe")) {
        observer.observe(iframe);
    }
});
