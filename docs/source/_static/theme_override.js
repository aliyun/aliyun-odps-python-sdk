$(function() {
    for (var hn = 1; hn <= 6; hn++) {
        $('.rubric-h' + hn).each(function (i, r) {
            $(r).replaceWith('<div class="rubric"><h' + hn + ' class="rubric-sub">' + $(r).html() + '</h' + hn + '></div>');
        });
    }
    $('p.rubric').each(function (i, r) {
        $(r).replaceWith('<div class="rubric"><h6 class="rubric-sub rubric-default">' + $(r).html() + '</h6></div>');
    });
    $('.rubric').each(function (i, r) {
        $(r).attr('id', 'rubric' + (i + 1));
        $(r).find('.rubric-sub').append('<a class="headerlink" href="#rubric' + (i + 1) + '">¶</a>');
    });
});
