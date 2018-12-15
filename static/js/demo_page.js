/**
 * Created by erdem on 9.05.17.
 */
function sendToServer(data, url,  callback) {
    console.log(url)
    console.log(data)
    console.log("=====================")
    $.ajax({
        type: 'POST',
        url: url,
        data: data,
        dataType: 'text',
        success: callback
    });
}



$(document).ready(function () {
    var active_query = $("#segmented").val().split(" ").map(function (e) {return [e]});
    var baseQuery = $("#segmented").val().split(" ")
    var language = $("#language_val").val();
    var locked = []
    function renderList(resultList) {
        var response = $.parseJSON(resultList);
        $("#combinations tr").not(function(){ return !!$(this).has('th').length; }).remove();
        $.each(response, function(i, item) {
            var link_element = '<a href="/query?lang='+language+'&query=' + item.phrase+'">';
            $('<tr>')
                .append( $('<td>').append($(link_element).text(item.phrase)),
                    $('<td>').text(item.score))
                .appendTo('#combinations');
        });
    }


    function renderSimilars(word) {
        var baseIndex = baseQuery.indexOf(word)
        $.get("/api/similars/all?language="+language+"&word="+word, function (data) {
            data = $.parseJSON(data);
            $.each(data, function(i, item) {
                $('<tr class="clickable-row similar_word" data-index='+baseIndex+' data-value='+item[0]+'>')
                    .append( $('<td>').text(item[0]),
                        $('<td>').text(item[1]))
                    .appendTo('#table'+word);
            });
            return sendToServer({data: JSON.stringify(active_query), locked: JSON.stringify(locked)},"/combinations?language="+ language, renderList)
        })
    }


    sendToServer({data: JSON.stringify(active_query)}, "/api/combinations?language="+ language, renderList);


    $('body').on('click', '.similar_word', function (){
        var row = $(this);
        var index = row.data("index");
        var value = row.data("value");
        if(row.hasClass("success")){
            active_query[index].splice(active_query[index].indexOf(value), 1);
            row.removeClass("success")
        } else {
            row.addClass("success")
            active_query[index].push(value);
        }
        $.ajax({
            type: 'POST',
            url: "/api/combinations?language="+ language,
            data:  {
                data: JSON.stringify(active_query)
            },
            dataType: 'text',
            success: renderList
        });
        sendToServer({data: JSON.stringify(active_query)}, "/api/combinations?language="+ language, renderList);
    });
    $("#updateQuery").click(function (e) {
        var query = $("#segmented").val().split(" ");

    });

    $(".checkbox").click(function (e) {
        var box = $(this)
        var word = box.data("word");
        active_query[baseQuery.indexOf(word)] =[word];
        if(box.is(":checked")){
            locked.push(word);
            $("#table"+word+" tr").not(function(){ return !!$(this).has('th').length; }).remove();
            return sendToServer({data: JSON.stringify(active_query), locked: JSON.stringify(locked)},"/api/combinations?language="+ language, renderList)
        }else {
            active_query[active_query.indexOf(word)] =[word];
            locked.splice(locked.indexOf(word), 1);
            renderSimilars(word)
        }


    });

});
