document.addEventListener("DOMContentLoaded", function () {
  const observer = new IntersectionObserver(
    function (entries, observer) {
      entries.forEach(function (video) {
        for (const source in video.target.children) {
          const videoSource = video.target.children[source];
          if (
            typeof videoSource.tagName === "string" &&
            videoSource.tagName === "SOURCE"
          ) {
            if (video.isIntersecting) {
              if (!videoSource.src) {
                videoSource.src = videoSource.dataset.src;
                video.target.load();
              } else {
                video.target.play();
              }
            } else {
              video.target.pause();
            }
          }
        }
      });
    },
    {
      threshold: 0.1,
    }
  );
  for (const video of document.getElementsByTagName("video")) {
    observer.observe(video);
  }
});
