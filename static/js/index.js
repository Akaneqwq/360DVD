document.addEventListener("DOMContentLoaded", function () {
    const observer = new IntersectionObserver(
        function (entries) {
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
                        }
                    }
                }
            });
        },
        {
            threshold: 0.1,
        }
    );

    function openVr(src) {
        const vrRoot = document.createElement("div");
        vrRoot.id = "vr-root";
        document.body.append(vrRoot);

        // 防止点击穿透
        vrRoot.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
        });

        // 防止滚动穿透
        const scrollTop = window.scrollY;
        document.body.dataset.scrollTop = scrollTop.toString();
        document.body.style.position = "fixed";
        document.body.style.top = `-${scrollTop}px`;
        document.body.style.bottom = "0";
        document.body.style.overflow = "hidden";

        const aSceneRoot = document.createElement("div");
        aSceneRoot.classList.add("vr-container");
        aSceneRoot.innerHTML = `
<a-scene vr-mode-ui="enabled: false" embedded>
    <a-assets>
        <video id="vr-video" loop playsinline muted autoplay src="${src}"></video>
    </a-assets>
    <a-videosphere src="#vr-video"></a-videosphere>
</a-scene>`;
        vrRoot.append(aSceneRoot);
        document.getElementById("vr-video").play();

        const closeBtn = document.createElement("div");
        closeBtn.ariaLabel = "close-vr";
        closeBtn.classList.add("close-btn");
        closeBtn.innerHTML = `
<svg t="1707812514995" class="icon" viewBox="0 0 1025 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4211" width="40" height="40">
    <path d="M513.344 0a512 512 0 1 0 0 1024 512 512 0 0 0 0-1024z m226.048 674.624l-54.528 56.896-171.52-164.928-171.392 164.928-54.592-56.896L456.576 512 287.36 349.312l54.592-56.768 171.392 164.8 171.52-164.8 54.528 56.768L570.176 512l169.216 162.624z" fill="#ffffff" p-id="4212"></path>
</svg>`;
        closeBtn.addEventListener("click", closeVr);
        vrRoot.append(closeBtn);
    }

    function closeVr() {
        // 恢复滚动
        const scrollTop = document.body.dataset.scrollTop;
        document.body.style.position = "unset";
        document.body.style.top = `-${scrollTop}px`;
        document.body.style.bottom = "0";
        document.body.style.overflow = "unset";
        document.documentElement.scrollTo(0, scrollTop);

        // 关闭 vr 展示
        const vrRoot = document.getElementById("vr-root");
        vrRoot.remove();
    }

    function isWx() {
        // 通过 ua 判断是否微信
        const ua = navigator.userAgent;
        return ua.indexOf("MicroMessenger") !== -1;
    }

    for (const elem of document.querySelectorAll(".video")) {
        try {
            observer.observe(elem.children[0]);

            const openVrElem = document.createElement("div");
            openVrElem.ariaLabel = "open-vr";
            openVrElem.classList.add("open-vr");
            elem.append(openVrElem);

            const ua = navigator.userAgent;
            // 微信内打开的页面不支持 aframe
            if (isWx()) continue;
            // 非微信场景，点击视频打开全景预览
            const sourceElem = elem.children[0].children[0];
            const src = sourceElem.dataset.vrsrc || sourceElem.dataset.src;

            openVrElem.addEventListener("click", () => {
                openVr(src);
            });
        } catch (e) {
            console.error(e);
        }
    }

    // 非微信中，提示可以点击视频打开全景预览
    const text = isWx()
        ? "Open in browser for better experience"
        : "Click any video to open VR preview";
    const toastElem = document.createElement("div");
    toastElem.id = "toast";
    document.body.append(toastElem);
    toastElem.innerHTML = text;

    toastElem.classList.add("showToast");
    // setTimeout(() => {
    //     toastElem.classList.remove("showToast");
    //     toastElem.classList.add("hideToast");
    //     toastElem.addEventListener("animationend", () => {
    //         toastElem.remove();
    //     });
    // }, 4000);
});
