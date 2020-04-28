package examples.a_mnist

interface Example {
    fun run()
}


fun <T> logTime(title:String = "",func: () -> T) :T {
    val start = System.currentTimeMillis()
    val invoked = func.invoke()
    val end = System.currentTimeMillis()
    println("Duration $title: ${end-start}ms")
    return invoked
}