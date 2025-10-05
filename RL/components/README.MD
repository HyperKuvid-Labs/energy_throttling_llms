# Components

The idea is something like using 
> Two Actors
> - Target actor
> - Fast updation actor

> One critic
> - Q-target critic

So the thing goes like this:
- the policy spits out an state, which is just one state in our case, with the three params, and if its in some boundary, we're gonna proceed.
- calculate the q of this state, and not still update the policy
- when the next one comes, we're gonna take a td error in hand and then update the policy
- so after some iteratons, the batch updates the target policy, and till now only the target policy gives out the value for q, we'll be using target policy over here, but updating the fast updating policy, this may be confusing, but when i code it up, it's more easier. 