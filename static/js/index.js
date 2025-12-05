window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }
    bulmaSlider.attach();


    // ==============================
    //  THUMBNAIL → MAIN VIDEO SWITCH
    // ==============================

    const mainVideoContainer = document.querySelector('.thumbnails');
    const mainVideo  = document.getElementById('main-video');
    const mainSource = document.getElementById('main-video-source');

    if (mainVideoContainer && mainVideo && mainSource) {
        const thumbs = mainVideoContainer.querySelectorAll('.thumb');

        thumbs.forEach(function(thumb) {
            thumb.addEventListener('click', function () {
                const newSrc = thumb.getAttribute('data-video');
                if (!newSrc || newSrc === mainSource.getAttribute('src')) return;

                // Update main video
                mainSource.setAttribute('src', newSrc);
                mainVideo.load();
                mainVideo.play();

                // Optional: active styling
                thumbs.forEach(t => t.classList.remove('is-active'));
                thumb.classList.add('is-active');
            });
        });
    }
    

    // ==============================
    //  THUMBNAIL → StableGrasp VIDEO SWITCH
    // ==============================
    const sgContainer = document.querySelector('.sg-thumbnails');
    const sgVideo  = document.getElementById('sg-video');
    const sgSource = document.getElementById('sg-video-source');

    if (sgContainer && sgVideo && sgSource) {
        const sgThumbs = sgContainer.querySelectorAll('.thumb');

        sgThumbs.forEach(function(thumb) {
            thumb.addEventListener('click', function () {
                const newSrc = thumb.getAttribute('data-video');
                if (!newSrc || newSrc === sgSource.getAttribute('src')) return;

                // Update sg video
                sgSource.setAttribute('src', newSrc);
                sgVideo.load();
                sgVideo.play();

                // Optional: active styling
                sgThumbs.forEach(t => t.classList.remove('is-active'));
                thumb.classList.add('is-active');
            });
        });
    }

})
