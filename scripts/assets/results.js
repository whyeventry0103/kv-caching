(function () {
  var tip = document.getElementById('tip');
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('mouseenter', function (e) {
      tip.textContent = el.getAttribute('data-tip');
      tip.style.opacity = '1';
    });
    el.addEventListener('mousemove', function (e) {
      var r = tip.getBoundingClientRect();
      var x = Math.min(e.clientX + 14, window.innerWidth - r.width - 8);
      tip.style.left = x + 'px';
      tip.style.top = (e.clientY - r.height - 12) + 'px';
    });
    el.addEventListener('mouseleave', function () { tip.style.opacity = '0'; });
  });
  var btn = document.querySelector('.themetoggle');
  btn.addEventListener('click', function () {
    var dark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  });
})();
